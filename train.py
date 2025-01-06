from ultralytics import YOLO
if __name__ == '__main__':
	# model = YOLO(rf"yolov8n-cls.pt")
	# model.train(cfg=rf"config.yaml")
	model = YOLO(rf"yolov8n-cls-resnet18.yaml")
	model.train(cfg=rf"config2.yaml")