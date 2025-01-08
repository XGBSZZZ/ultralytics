from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(rf"custom_data/yolov8n-cls.pt")
    # model.train(cfg=rf"custom_data/config.yaml")
    model = YOLO(rf"resnet18.yaml")
    model.train(cfg=rf"custom_data/config2.yaml")
