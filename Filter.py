#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import os

# ################################### ##### Yolo 모델이 없을 시 다운받는 코드#######
# try:
#     loadmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#     torch.save(loadmodel,'./FILTER(YOLO)/yolov5model.pth')
# except:
#     None
# torch_model = torch.load('./FILTER(YOLO)/yolov5model.pth')
# torch_model.to('cuda' if torch.cuda.is_available() else 'cpu')
# yolo_model = torch.load(,path= './FILTER(YOLO)/best.pt')

####### custom############
torch_model = torch.hub.load('./FILTER(YOLO)', 'custom', path= './FILTER(YOLO)/yolov5s.pt', source='local')
torch_model.to('cuda' if torch.cuda.is_available() else 'cpu')

def yolo(frame):
    global stop_person
    global stop_car
    global stop_kick
    global stop_motorcycle
    global stop_bicycle
    global count
    OBJnames = []
    result = torch_model(frame)
    labels, cord = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
    yolo_name = torch_model.names
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.75:
            OBJnames.append(yolo_name[int(labels[i])])
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            white = (255,255,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), white, 3)
            cv2.putText(frame, (yolo_name[int(labels[i])]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 3)
        else:
            pass

    return frame