#encoding=utf-8

from app.yolov5 import *
from app.paddleclas import *
import cv2

print(cv2.__version__)

print(u"opencv-python version == {}".format(cv2.__version__))
frame = cv2.imread('./images/crowdpose_100024.jpg')
yolov5_net = yolov5('kapao_s_coco.onnx')
paddle_net = paddleclas.PaddleClas(model_name='person_attribute')
preson_attrs = dectect_person_attr_from_frame(yolov5_net,paddle_net,frame)
for item in preson_attrs:
    print(item[0])
