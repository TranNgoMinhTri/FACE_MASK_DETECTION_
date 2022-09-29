import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCDispatcher

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

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov4.weights', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/face_mask.names', help='*.cfg path')
opt = parser.parse_args()



out, source, weights, view_img, save_txt, imgsz, cfg, names = \
opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

vid_path, vid_writer = None, None
dataset = LoadImages(source, img_size=imgsz, auto_size=64)
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

model = Darknet(cfg, imgsz).cuda()
if half:
    model.half()
try:
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
except:
    load_darknet_weights(model, weights[0])

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
  # run once
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)



pred = model(img, augment=opt.augment)[0]
pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

names = load_classes(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
save_img = False
view_img =True
for i, det in enumerate(pred):  # detections per image
   
    p, s, im0 = path, '', im0s
    save_path = str(Path(out) / Path(p).name)
    txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
    s += '%gx%g ' % img.shape[2:]  # print string
    # print(s)
    # print(img.shape)
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, names[int(c)])  # add to string
    print(s)
    save_txt =  True
    # Write results
    for *xyxy, conf, cls in det:
        # if save_txt:  # Write to file
        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #     with open(txt_path + '.txt', 'a') as f:
        #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

       
        label = '%s %.2f' % (names[int(cls)], conf)
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    
    cv2.imshow("result", im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", im0)

