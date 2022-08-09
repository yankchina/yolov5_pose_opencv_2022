from app import yolov5
import cv2 as cv

images = cv.imread('./images/crowdpose_100024.jpg')
yolov5 = yolov5()