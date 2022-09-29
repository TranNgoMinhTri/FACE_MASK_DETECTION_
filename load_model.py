import argparse
import os
import platform
from re import T
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

# # def detect(save_img=False):
# #     out, source, weights, view_img, save_txt, imgsz, cfg, names = \
# #         opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
# #     webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

# #     # Initialize
# #     device = select_device(opt.device)
# #     if os.path.exists(out):
# #         shutil.rmtree(out)  # delete output folder
# #     os.makedirs(out)  # make new output folder
# #     half = device.type != 'cpu'  # half precision only supported on CUDA

# #     # Load model
# #     model = Darknet(cfg, imgsz).cuda()
# #     try:
# #         model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
# #         #model = attempt_load(weights, map_location=device)  # load FP32 model
# #         #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
# #     except:
# #         load_darknet_weights(model, weights[0])
# #     model.to(device).eval()
# #     if half:
# #         model.half()  # to FP16
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov4.weights', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--cfg', type=str, default='models/yolov4.cfg', help='*.cfg path')
#     parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
#     opt = parser.parse_args()
#     print(opt)
#     # detect()
