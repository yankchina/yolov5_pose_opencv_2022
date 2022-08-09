#encoding=utf-8
from app.yolov5 import *
import cv2 
import time

start_time = time.time()
weight_name = u"kapao_s_coco.onnx"
print(u"opencv-python version == {}".format(cv2.__version__))
net = yolov5(u"weights/{}".format(weight_name))

load_time = time.time() - start_time
print(u"load time == {}".format(load_time))

video_file_name = "demo.mp4"
out_video_file_name = "out.mp4"

cap = cv2.VideoCapture(video_file_name)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_video_file_name,fourcc,fps,(width,height))

if(cap.isOpened() == False):
    print("Error opening video stream or file")
else:
    frame_index = 0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            result_frame = net.detect(frame)
            out.write(result_frame)
        frame_index += 1
        print(frame_index)
        
end_time = time.time() - start_time - load_time
cap.release()
out.release()
print("Done in {} seconds".format(end_time))
            