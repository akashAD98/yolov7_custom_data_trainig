import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import time 

weights = "/mnt/c/Users/user/Desktop/yolov7/runs/train/exp12/weights/best.pt"
video_path = "/mnt/c/Users/user/Desktop/DeepLearningStuff/driving.mp4"
cap = cv2.VideoCapture(video_path)
device =  0 
device = select_device(str(device))
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())

imgsz = 640
trace = False
half = True

conf_thres = 0.25
iou_thres = 0.25


if trace:
        model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16
    

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once for warmup

while True:
    _,im0 = cap.read()
    img = letterbox(im0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time.time()
    with torch.no_grad():
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False) #for filtering change classes
    t2 = time.time()
    print(f"FPS:{1 /(t2-t1):.2f}")
    for i, det in enumerate(pred):  # detections per image
            
        if len(det):
                # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):     
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)



    cv2.imshow("Frame", im0)
    ch = cv2.waitKey(1)  # 1 millisecond
    if ch == ord("q") : break 