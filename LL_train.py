# https://blog.csdn.net/magic_ll/article/details/135102882
from ultralytics import YOLO
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

root = os.getcwd()
# 配置文件路径
config_yaml = os.path.join(root, "zzz_data/config.yaml")
config_final_yaml = os.path.join(root, "zzz_data/config_fine.yaml")
name_pretrain = os.path.join(root, "zzz_data/yolov8n-cls.pt")
# 原始训练路径
path_train = os.path.join(root, "runs/pruning/source")
name_train = os.path.join(path_train, "weights/last.pt")
# 约束训练路径、剪枝模型文件
path_constraint_train = os.path.join(root, "runs/pruning/dataset_Constraint")
name_prune_before = os.path.join(path_constraint_train, "weights/last.pt")
name_prune_after = os.path.join(path_constraint_train, "weights/last_prune.pt")
# 微调路径
path_fineturn = os.path.join(root, "runs/pruning/dataset_finetune")


def else_api():
    path_data = ""
    path_result = ""
    model = YOLO(name_pretrain)
    metrics = model.val()  # evaluate model performance on the validation set
    model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
    model.predict(path_data, device="0", save=True, show=False, save_txt=True, imgsz=640, save_conf=True,
                  name=path_result, iou=0.5)  # 这里的imgsz为高宽


def step1_train(L1_regulation=1e-2):
    model = YOLO(name_pretrain)
    model.train(cfg=config_yaml, name=path_train, L1_regulation=L1_regulation)  # train the model


# 2024.3.4添加【amp=False】
def step2_Constraint_train():
    model = YOLO(name_train)
    model.train(cfg=config_yaml, amp=False, name=path_constraint_train)  # train the model


def step3_pruning():
    from LL_pruning import do_pruning
    do_pruning(name_prune_before, name_prune_after)


def step4_finetune():
    model = YOLO(name_prune_after)  # load a pretrained model (recommended for training)
    model.train(cfg=config_final_yaml, name=path_fineturn)  # train the model


if __name__ == "__main__":
    step1_train()
    # step2_Constraint_train()
    # step3_pruning()
    # step4_finetune()
