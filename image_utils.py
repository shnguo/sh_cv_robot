import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from io import BytesIO

def base642image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img


def image2base64(image):
    img = cv2.imencode('.png', image)[1]
    img_base64 = str(base64.b64encode(img))[2:-1]
    return img_base64


def image2base64_new(img):
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    # im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_arr)
    return im_b64

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    base_path = Path(__file__).resolve().parent
    fontStyle = ImageFont.truetype(
        f"{base_path}/fonts/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def base64_img(base64str):
    return Image.open(BytesIO(base64.b64decode(base64str)))

def img_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def img_base64_new(image):
    encoded_image = base64.b64encode(image.tobytes())
    return str(encoded_image)
