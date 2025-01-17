# https://blog.csdn.net/magic_ll/article/details/135102882
from ultralytics.zzz_ultralytics.zzz_L1_pruning import do_pruning
from ultralytics import YOLO
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

root = os.getcwd()
# 配置文件路径
config_yaml = os.path.join(root, "config.yaml")
config_finetune_yaml = os.path.join(root, "config.yaml")
name_pretrain = os.path.join(root, "yolov8n-cls.pt")
# 原始训练路径
path_train = os.path.join(root, "runs/pruning/source")
name_train = os.path.join(path_train, "weights/last.pt")
# 约束训练路径、剪枝模型文件
path_constraint_train = os.path.join(root, "runs/pruning/zzz_Constraint")
name_prune_before = os.path.join(path_constraint_train, "weights/last.pt")
name_prune_after = os.path.join(path_constraint_train, "weights/last_prune.pt")
# 微调路径
path_fineturn = os.path.join(root, "runs/pruning/zzz_finetune")


def else_api():
    path_data = ""
    path_result = ""
    model = YOLO(name_pretrain)
    metrics = model.val()  # evaluate model performance on the validation set
    model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
    model.predict(path_data, device="0", save=True, show=False, save_txt=True, imgsz=640, save_conf=True,
                  name=path_result, iou=0.5)  # 这里的imgsz为高宽


def step1_train(cfg, model_yaml=None, model_path=None):
    if model_yaml is None and model_path is None:
        raise "model_yaml is None and model_path is None"
    elif model_yaml is not None and model_path is not None:
        model = YOLO(model_yaml).load(model_path)
    else:
        model = YOLO(model_yaml if model_yaml is not None else model_path)

    model.train(cfg=cfg, name=path_train, L1_regulation=-1)  # train the model


# 2024.3.4添加【amp=False】
def step2_Constraint_train(cfg, model_yaml=None, model_path=None, L1_regulation=1e-2):
    if model_yaml is None and model_path is None:
        raise "model_yaml is None and model_path is None"
    elif model_yaml is not None and model_path is not None:
        model = YOLO(model_yaml).load(model_path)
    else:
        model = YOLO(model_yaml if model_yaml is not None else model_path)

    model.train(cfg=cfg, amp=False, exist_ok=True, name=path_constraint_train,
                L1_regulation=L1_regulation)  # train the model


def step3_pruning(prune_radio=0.8):
    do_pruning(name_prune_before, name_prune_after, prune_radio)


def step4_finetune(cfg, L1_finetune=True):
    model = YOLO(name_prune_after)  # load a pretrained model (recommended for training)
    model.train(cfg=cfg, exist_ok=True, name=path_fineturn, L1_regulation=-1,
                L1_finetune=L1_finetune)  # train the model
    model.export(format="onnx")


def zzz_train(normal_train=True, normal_Constraint_model_yaml=None, normal_Constraint_model_path=None, cfg_normal=None,
              cfg_finetune=None, prune_radio=0.8):
    if normal_train:
        step1_train(cfg_normal, normal_Constraint_model_yaml, normal_Constraint_model_path)
    step2_Constraint_train(cfg_normal, normal_Constraint_model_yaml, normal_Constraint_model_path)
    step3_pruning(prune_radio)
    step4_finetune(cfg_finetune)

# if __name__ == "__main__":
#     step1_train()
#     step2_Constraint_train()
#     step3_pruning()
#     step4_finetune()
