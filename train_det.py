from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO(rf"custom_data/config_det.yaml").load(rf"custom_data/yolov8n.pt")
	model.train(cfg=rf"config.yaml")