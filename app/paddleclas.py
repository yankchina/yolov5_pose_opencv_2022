import paddleclas
import os

def crop_image_with_box(src_img,box):
    left = box[1]
    top = box[2]
    right = left + box[3]
    bottom = top + box[4]
    result_image = src_img[top:bottom, left:right]
    return result_image

def dectect_person_attr_from_frame(yolov5_net,paddle_net,frame):
    person_attrs = []
    ## 检测图片中的所有人员
    yolov5_net.detect(frame)
    boxes = yolov5_net.boxes
    for box in boxes:
        img2 = crop_image_with_box(frame,box)
        result = paddle_net.predict(input_data = img2)
        person_attrs.append(next(result))
    ## 逐个检测每个图片中的数据信息
    return person_attrs


def get_paddleclass_net(model_name="person_attribute"):
    ##TODO: load model from file
    model_file_path = os.path.join(os.path.abspath(__file__))
    paddle_net = paddleclas.PaddleClass(model_name=model_name)
    return paddle_net