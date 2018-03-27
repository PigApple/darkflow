from darkflow.defaults import argHandler #Import the default arguments
import os
from darkflow.net.build import TFNet
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

import tf_parameter_mgr
from cfgyolo import updateCfg
from monitor_cb import CMonitor

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('--job_name', dest='job_name',
                        help='One of "ps", "worker"',required=True,type=str)
    parser.add_argument('--ps_hosts', dest='ps_hosts',
                        help='Comma-separated list of hostname:port for the parameter server jobs',default=None,type=str)
    parser.add_argument('--worker_hosts', dest='worker_hosts',
                        help='Comma-separated list of hostname:port for the worker jobs',required=True,type=str)
    parser.add_argument('--task_id', dest='task_id',
                        help='Task ID of the worker/replica running the training',required=True,type=int)

    parser.add_argument('--train_dir', dest='train_dir',
                        help='Directory where to write event logs and checkpoint',required=True,type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)


    parser.add_argument('--yolover', dest='yolover',
                        help='Yolo version',
                        default=2, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='default yolov2 configure file',
                        default='./cfg/yolov2.cfg', type=str)

    args = parser.parse_args()
    return args

def _get_other_format_weights(weights):
    files = []
    for file in os.listdir(weights):
        files.append(os.path.join(weights,file))
    if len(files) == 0:
        return None
    else:
        return files[0]

def train_model(FLAGS, ckpt, server):

    tfnet = TFNet(FLAGS)
    tfnet.framework.loss(tfnet.out)
    tfnet.say('Building {} train op'.format(tfnet.meta['model']))
    global_step = tf.contrib.framework.get_or_create_global_step() 
    
    lr = tf.train.exponential_decay(tfnet.FLAGS.lr, global_step,
                                10000, tfnet.FLAGS.lrDecay, staircase=True)
    tfnet.train_op = tf_parameter_mgr.getOptimizer(lr).minimize(tfnet.framework.loss, global_step=global_step)

    batches_generator = tfnet.framework.shuffle()
    test_batches_generator = tfnet.framework.shuffle_test()
    loss_ph = tfnet.framework.placeholders
    loss_op = tfnet.framework.loss
    
    variables_to_restore_all = []
    for var in tf.global_variables():#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print var
        print var.name
        if not var.op.name.startswith('global_step'):
            variables_to_restore_all.append(var)
    saver = tf.train.Saver(var_list=variables_to_restore_all)

    LOG_DIR = os.path.join(FLAGS.train_dir,'log')
    if FLAGS.task_index == 0: #master
        monitor = CMonitor(LOG_DIR, tf_parameter_mgr.getTestInterval(), tf_parameter_mgr.getMaxSteps())
        graph = tf.get_default_graph()
        all_ops = graph.get_operations()
        for op in all_ops:
            if op.type == 'VariableV2':
                print 'var name:', op.name
            elif op.type == 'Relu':
                print 'Relu:', op.name
        
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            output_tensor = graph.get_tensor_by_name(var.name)
            var_splits = var.name.split('/')
            if var_splits[1].startswith('kernel'):
                print 'this is weight variable:', var.name
                monitor.SummaryHist("weight", output_tensor, var_splits[0])
                monitor.SummaryNorm2("weight", output_tensor, var_splits[0])
            if var_splits[1].startswith('biases'):
                print 'this is bias variable:', var.name
                monitor.SummaryHist("bias", output_tensor, var_splits[0])  
            if var_splits[1].startswith('gamma'):
                print 'this is batch norm variable:', var.name   
                monitor.SummaryHist("activation", output_tensor, var_splits[0])

        monitor.SummaryScalar("train loss", loss_op)
        monitor.SummaryGradient("weight", loss_op)
        monitor.SummaryGWRatio()
        merged = tf.summary.merge_all()
        summaryWriter = tf.summary.FileWriter(LOG_DIR)      

    class _CKHook(tf.train.SessionRunHook):
        def __init__(self):
            self._next_trigger_step = FLAGS.test_interval
            self._trigger = False
        
        def before_run(self, run_context):
            args = {'global_step': global_step}
            if self._trigger:
                self._trigger = False
                args['summary'] = merged
            return tf.train.SessionRunArgs(args)
        
        def after_run(self, run_context, run_values):

            u_gs = run_values.results['global_step']
            if u_gs >= self._next_trigger_step:
                self._trigger = True
                self._next_trigger_step += FLAGS.test_interval

            summary = run_values.results.get('summary', None)
            if summary is not None:
                summaryWriter.add_summary(summary, u_gs)
                               
                test_loss = []
                
                for i in range(FLAGS.test_iter):
                    test_x_batch, test_datum = next(test_batches_generator)
                    feed_dict = {
                                loss_ph[key]: test_datum[key] 
                                for key in loss_ph }
                    feed_dict[tfnet.inp] = test_x_batch
                    feed_dict.update(tfnet.feed)  
                                          
                    fetches = [loss_op]                                     
                    loss = run_context.session.run(fetches, feed_dict)
                    test_loss += [loss]
                                    
                form = '>>>> test step {} - loss {}'
                tfnet.say(form.format(u_gs, np.mean(test_loss)))
                test_summary = tf.Summary(value=[tf.Summary.Value(tag='test loss', simple_value=np.mean(test_loss))])
                summaryWriter.add_summary(test_summary, u_gs)
                                                
            if u_gs >= FLAGS.maxSteps:
                #for node in run_context.session.graph_def.node:
                #    node.device = ""
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.weightPrefix)
                saver.save(run_context.session, checkpoint_path, global_step=u_gs)

    class _RestoreHook(tf.train.SessionRunHook):
        def after_create_session(self, session, coord):

            if ckpt is not None:
               print "Restore pre-trained checkpoint detail:", ckpt
               saver.restore(session, ckpt)
            elif FLAGS.load != -1:
               print "Try to restore from other checkpoint type here...."
            else:
               print "No pre-triained weith file found"
                    
    
    hooks = [tf.train.StopAtStepHook(last_step = FLAGS.maxSteps), _RestoreHook()]
    with tf.train.MonitoredTrainingSession(master = server.target,
                                           is_chief = (FLAGS.task_index == 0),
                                           hooks = hooks,
                                           chief_only_hooks = [_CKHook()]
                                           ) as mon_sess:
        while not mon_sess.should_stop():
            x_batch, datum = next(batches_generator)
            feed_dict = {
                loss_ph[key]: datum[key] 
                    for key in loss_ph }
            feed_dict[tfnet.inp] = x_batch                        
            feed_dict.update(tfnet.feed)                                                
            fetches = [tfnet.train_op, loss_op, global_step]

            _, loss, g_s = mon_sess.run(fetches, feed_dict)

            form = 'step {} - loss {}'
            tfnet.say(form.format(g_s+1, loss))            

if __name__ == '__main__':
    args = parse_args()

    FLAGS = argHandler()
    FLAGS.setDefaults()
    
    def _get_dir(dirs):
      for d in dirs:
        this = os.path.abspath(os.path.join(os.path.curdir, d))
        if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])  #some of thome is not neccessary
    
    FLAGS.distributed = True
    FLAGS.isTextDataSet = True
    FLAGS.batch = tf_parameter_mgr.getTrainBatchSize()
    FLAGS.maxSteps = tf_parameter_mgr.getMaxSteps()
    
    FLAGS.lr = tf_parameter_mgr.getBaseLearningRate()
    FLAGS.lrDecay = tf_parameter_mgr.getLearningRateDecay()  
    FLAGS.test_interval=tf_parameter_mgr.getTestInterval()
    FLAGS.test_iter = 2 
    
    FLAGS.trainSet = tf_parameter_mgr.getTrainData()[0]    #eg. ../../train.txt
    FLAGS.testSet = tf_parameter_mgr.getTestData()[0]                  #eg. ../../test.txt
    
    FLAGS.dataPath = os.path.join(os.path.split(os.path.abspath(FLAGS.trainSet))[0], "..", "..")
    FLAGS.labels = os.path.join(os.path.split(os.path.abspath(FLAGS.trainSet))[0], 'labels.txt')

    FLAGS.annotation = os.path.join(FLAGS.dataPath, 'Annotations')
    FLAGS.dataset = os.path.join(FLAGS.dataPath, 'JPEGImages')

    #with tf.Graph().as_default(), tf.device('/cpu:0'):
    worker_hosts = args.worker_hosts.split(',')
    print 'worker_hosts:', worker_hosts
    if args.ps_hosts is not None:
        ps_hosts = args.ps_hosts.split(',')
        print 'ps_hosts:', ps_hosts
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    else:
        cluster = tf.train.ClusterSpec({"worker": worker_hosts})

    server = tf.train.Server(
        cluster,
        job_name = args.job_name,
        task_index=args.task_id)

    if args.job_name == 'ps':
        server.join()

    is_chief = (args.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(args.train_dir):
            tf.gfile.DeleteRecursively(args.train_dir)
        tf.gfile.MakeDirs(args.train_dir)    
    
    try: FLAGS.load = int(FLAGS.load)  #not nceccessary since we will have args.weights indicating the weight file path 
    except: pass
    
    FLAGS.train_dir = args.train_dir
    FLAGS.task_index = args.task_id
    FLAGS.pretrained_model = args.pretrained_model
    
    
    if not gfile.Exists(args.cfg_file):
        CLASSES = [line.rstrip('\n') for line in open(FLAGS.labels).readlines()]
        if args.yolover == 1:
            updateCfg('./cfg/template/yolov1.cfg', args.cfg_file, len(CLASSES), yolo_version=1)
        else:
            updateCfg('./cfg/template/yolov2.cfg', args.cfg_file, len(CLASSES), yolo_version=2)
    FLAGS.model = args.cfg_file
    
    FLAGS.weightPrefix = os.path.basename(args.cfg_file).split('.')[0]
    print 'FLAGS.weightPrefix: ', FLAGS.weightPrefix 
    
    ckpt = None
    FLAGS.load = -1
    if FLAGS.pretrained_model is not None:
        ckpt=tf.train.latest_checkpoint(FLAGS.pretrained_model)
    
        if ckpt is None:
          FLAGS.load = _get_other_format_weights(FLAGS.pretrained_model)
    
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args.task_id, cluster=cluster)):
        train_model(FLAGS, ckpt, server)
    
    
    
    
