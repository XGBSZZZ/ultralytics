from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect

common_path = rf"runs/detect/train15/weights/"
yolo = YOLO(rf"{common_path}/last.pt")  # 第一步约束训练得到的pt文件

model = yolo.model
ws = []
bs = []

for _, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        w = m.weight.abs().detach()
        b = m.bias.abs().detach()
        ws.append(w)
        bs.append(b)

factor = 0.1  # 通道保留比率

ws = torch.cat(ws)
threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
print(threshold)


def _prune(c1, c2):
    wet = c1.bn.weight.data.detach()
    bis = c1.bn.bias.data.detach()
    _list = []
    _threshold = threshold
    while len(_list) < 8:
        _list = torch.where(wet.abs() >= _threshold)[0]
        _threshold = _threshold * 0.5
    i = len(_list)
    c1.bn.weight.data = wet[_list]
    c1.bn.bias.data = bis[_list]
    c1.bn.running_var.data = c1.bn.running_var.data[_list]
    c1.bn.running_mean.data = c1.bn.running_mean.data[_list]
    c1.bn.num_features = i
    c1.conv.weight.data = c1.conv.weight.data[_list]
    c1.conv.out_channels = i
    if c1.conv.bias is not None:
        c1.conv.bias.data = c1.conv.bias.data[_list]
    if not isinstance(c2, list):
        c2 = [c2]
    for item in c2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            conv.in_channels = i
            conv.weight.data = conv.weight.data[:, _list]


def prune(m1, m2):
    if isinstance(m1, C2f):
        m1 = m1.cv2
    if not isinstance(m2, list):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1
    _prune(m1, m2)


for _, m in model.named_modules():
    if isinstance(m, Bottleneck):
        _prune(m.cv1, m.cv2)

for _, p in yolo.model.named_parameters():
    p.requires_grad = True

yolo.export(format="onnx")  # 导出为onnx文件
# yolo.train(data="dataset.yaml", epochs=100) # 剪枝后直接训练微调

torch.save(yolo.ckpt, rf"{common_path}/last_prune.pt")
print("done")