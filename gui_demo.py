#encoding=utf-8

from app.yolov5 import *
import cv2 
import time


def demo_detect(source_image,weight_name):
    print(weight_name)
    start_time = time.time()
    net = yolov5(u"weights/{}".format(weight_name))
    load_time = time.time() - start_time
    result_image = net.detect(source_image)
    detect_time = time.time() - start_time - load_time
    cv2.imwrite('./images/{}_result.jpg'.format(weight_name), result_image)
    print(u"{0} load time: {1:.2f}s, detect time: {2:.2f}s".format(weight_name, load_time, detect_time))

print(u"opencv-python version == {}".format(cv2.__version__))

src_image = cv2.imread('./images/crowdpose_100024.jpg')

# demo_detect(src_image,'kapao_s_coco.onnx')
# demo_detect(src_image,'kapao_l_coco.onnx')
# demo_detect(src_image,'kapao_m_coco.onnx')
# demo_detect(src_image,'kapao_s_crowdpose.onnx')
# demo_detect(src_image,'kapao_l_crowdpose.onnx')
demo_detect(src_image,'kapao_m_crowdpose.onnx')