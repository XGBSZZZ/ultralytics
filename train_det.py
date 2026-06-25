from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml").load(rf"zzz_data/yolov8n.pt")
    model.train(cfg=rf"zzz_data/config_det.yaml",
                zzzhide=0.5,
                zzzrotate=1.0,
                fliproi=1.0, fliproi_names=["SCREW", "T_MOS_HS", "R_MOS_HS", "M2_80", "M2_HS", "PCH_HS", "DDR", "BATTERY", "RM", "SN"])
