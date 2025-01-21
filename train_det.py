from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(rf"yolov8n.yaml").load("zzz_data/yolov8n.pt")
    model.train(cfg=rf"zzz_data/config_det.yaml")
