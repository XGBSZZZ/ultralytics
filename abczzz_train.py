# https://blog.csdn.net/magic_ll/article/details/135102882
from ultralytics import zzz_train
import os

root = os.getcwd()
# 配置文件路径
config_yaml = os.path.join(root, "zzz_data/config.yaml")
config_finetune_yaml = os.path.join(root, "zzz_data/config.yaml")
name_pretrain = os.path.join(root, "zzz_data/yolov8n-cls.pt")

if __name__ == "__main__":
    zzz_train(True, None, name_pretrain, config_yaml, config_finetune_yaml, 0.5)
