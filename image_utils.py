import base64
import cv2
import numpy as np


def base642image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


def image2base64(image):
    img = cv2.imencode('.jpg', image)[1]
    img_base64 = str(base64.b64encode(img))[2:-1]
    return img_base64
