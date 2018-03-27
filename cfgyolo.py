
YOLOv1_GRID = 7
YOLOv1_BBOX = 2
YOLOv2_NUM = 5

def updateCfg(template_cfg, cfg_file, num_classes, yolo_version=2):
    
    def _parse(l, i = 1):
        return l.split('=')[i].strip()

    with open(template_cfg, 'rb') as f:
        lines = f.readlines()
    
    newfile_writer = open(cfg_file, 'wt')
    layer = []
    for line in lines:
        line = line.strip()
        line = line.rstrip('\n')
        
        if line.startswith('['):
            islast = line.startswith('[region]') or line.startswith('[detection]')
            if len(layer) > 0:
                print 'layer', layer
                for cfg in layer:
                    if islast and cfg.startswith('filters'):
                        
                        newfile_writer.write('filters={:d}\n'.format(calculateOutSize(num_classes, yolo_version)))
                    else:
                        newfile_writer.write(cfg + '\n')
                layer = []
            
        layer.append(line)        
    
    for cfg in layer:
        if cfg.startswith('classes'):
            newfile_writer.write('classes={:d}\n'.format(num_classes))
        else:
            newfile_writer.write(cfg + '\n')

    newfile_writer.close()

def calculateOutSize(num_classes, yolo_version=2):
    if yolo_version == 1:
        return YOLOv1_GRID*YOLOv1_GRID(YOLOv1_BBOX*5 + num_classes)
    if yolo_version == 2:
        return YOLOv2_NUM*(num_classes + 5)

if __name__ == '__main__':
    updateCfg('./cfg/yolo.cfg', 'yolov2.cfg', 6)
    