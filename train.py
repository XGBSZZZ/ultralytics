from ultralytics import YOLO, step3_pruning

if __name__ == '__main__':
    model = YOLO("yolov8n-cls.yaml").load(rf"zzz_data/yolov8n-cls.pt")
    model.train(cfg=rf"zzz_data/config.yaml")
    # model = YOLO(rf"resnet18.yaml")
    # model.train(cfg=rf"zzz_data/config2.yaml")
    # step3_pruning(0.05)
