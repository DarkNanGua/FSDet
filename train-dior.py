import os

from ultralytics.nn.modules.distill import MultiScaleDistillLoss
os.environ['PYTHONHASHSEED'] = str(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np
import torch
from ultralytics import YOLO
import argparse

import warnings



# 忽略所有关于torch使用确定性算法的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.use_deterministic_algorithms.*")

ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
                                            # 数据集
    parser.add_argument('--data', type=str, default= 'ultralytics/cfg/datasets/DIOR.yaml', help='dataset.yaml path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz','--img', type=int, default=640, help='inference size (pixels)')
                                        # 模型
    parser.add_argument('--config', type=str, default= 'ultralytics/cfg/models/v8/yolov8_propose_dysample_sk_3H.yaml', help='model path(s)')
    parser.add_argument('--resume', type=bool, default= False, help='resume?True or False')
    parser.add_argument('--premodel', type=str, default= 'premodel/yolov8s.pt', help='load pretain')
    # parser.add_argument('--premodel', type=str, default= 'output_dir/cocco/yolov8s5/weights/last.pt', help='load pretain')
    # parser.add_argument('--premodel', type=str, default= '', help='load pretain')
    # parser.add_argument('--freeze', type=int, default= 0, help='freeze')
                                        # 保存目录
    parser.add_argument('--project', default= 'output_dir/xiaorong/dior', help='save to project/name')
    parser.add_argument('--name', default='c-interpiou', help='save to project/name')
                                        # 训练参数
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--seed', type=int , default=42, help='random seed') # 随机种子
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=12, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    opt = parser.parse_args()
    return opt

def set_seed(seed=0):
    if(seed=='' or seed==0 or seed==None):
        return
    # torch.backends.cudnn.enabled = True  # pytorch 使用CUDANN 加速，即使用GPU加速
    torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
    torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
    torch.manual_seed(seed)  # 为当前CPU 设置随机种子
    # torch.cuda.manual_seed(seed)  # 为当前的GPU 设置随机种子
    # torch.cuda.manual_seed_all(seed)  # 当使用多块GPU 时，均设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

for _ in range(3):
    set_seed(42)
    a = torch.rand(3)
    print(a)
    
if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    set_seed(opt.seed)
    args = {
        "data": ROOT + opt.data,
        "epochs": opt.epochs,
        "resume": opt.resume,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": ROOT + opt.project,
        "name": opt.name,
        "imgsz": opt.imgsz, 
        "seed": opt.seed,
    }
    model_conf = ROOT + opt.config
    model_pretain = ROOT + opt.premodel
    if opt.resume == True:
        task_type = {
            "train": YOLO(model_pretain).train(resume=True),
            # "val": YOLO(model_pretain).val(resume=True),
            # "test": YOLO(model_pretain).val(resume=True),
        }
    elif opt.premodel != '':
        task_type = {
            "train": YOLO(model_conf).load(model_pretain).train(**args),
            # "val": YOLO(model_conf).val(**args),
            # "test": YOLO(model_conf).val(**args),
        }
    else:
        task_type = {
            "train": YOLO(model_conf).train(**args),
            # "val": YOLO(model_conf).val(**args),
            # "test": YOLO(model_conf).val(**args),
        }
    task_type.get(task)

