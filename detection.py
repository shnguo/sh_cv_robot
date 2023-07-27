import json
import os.path
import allspark
import torch
from torchvision import transforms
import cv2
from image_utils import image2base64, base642image
import argparse
import socket
from ultralytics import YOLO
from resnet.net import *
from PIL import Image
import numpy as np

torch.backends.cudnn.enabled = False


class Processor(allspark.BaseProcessor):
    def __init__(self, worker_threads=8, io_threads=8, worker_processes=1, host_number='9613',
                 model_path='person.pt', model_type=None, yolo_version='yolov5s',confidence=0.5):
        super().__init__(worker_threads=worker_threads, io_threads=io_threads, worker_processes=worker_processes, endpoint=f'0.0.0.0:{host_number}')
        self.model = None
        self.model_path = os.path.join('models/' + yolo_version, model_path)
        self.model_type = model_type
        self.yolo_version = yolo_version
        self.confidence = confidence

    def initialize(self):
        # 加载模型，只运行一次
        if 'yolov5' in self.yolo_version:
            self.model = torch.hub.load('../yolov5', 'custom', path=self.model_path, source='local', force_reload=True)
        elif self.yolo_version=='yolov8':
            self.model = YOLO(self.model_path)
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.model = torch.load(self.model_path)
            self.model.eval()
            # self.nonlinear = nn.Softmax(dim=1)

    def process(self, item):
        # 调用模型推理方法，每调用一次会运行一次
        try:
            item = json.loads(item.decode("utf-8").strip())
            img_raw = item['img_raw']
            image = base642image(img_raw)
            # print(image.shape)

            res = {"img": None, 'detection': None}

            flag = False
            label_res = []

            # 推理
            if 'yolov5' in self.yolo_version:
                results = self.model(image)
                outs = results.pandas().xyxy[0]
                for _, row in outs.iterrows():
                    if self.model_type == 'person' and row['class'] != 0:
                        continue
                    if float(row['confidence']) < self.confidence:
                        continue
                    results.print()
                    flag = True
                    cur_ = [row['class'], float('%.2f' % row['confidence']), int(row['xmin']),
                            int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                    cur_ = list(map(str, cur_))
                    label_res.append('_'.join(cur_))
                    cv2.rectangle(image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])),
                                  (0, 255, 0),
                                  thickness=2)
            elif self.yolo_version=='yolov8':
                # 每次只推理一张图像
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                results = self.model.predict(image)
                result = results[0].numpy() if self.model.device == 'cpu' else results[0].cpu().numpy()
                xyxys, clss, confs = result.boxes.xyxy, result.boxes.cls, result.boxes.conf
                for xyxy, cls, conf in zip(xyxys, clss, confs):
                    if self.model_type == 'person' and int(cls) != 0:
                        continue
                    if float(conf) < self.confidence:
                        continue
                    flag = True
                    cur_ = [cls, float('%.2f' % conf), int(xyxy[0]),
                            int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    cur_ = list(map(str, cur_))
                    label_res.append('_'.join(cur_))
                    # cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                    #               (0, 255, 0),
                    #               thickness=2)
            elif self.yolo_version=='resnet50':
                image_pil = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                nonlinear = nn.Softmax(dim=1)
                sample = self.transforms(Image.fromarray(image_pil)).unsqueeze(0)
                outputs = nonlinear(self.model(sample)).tolist()[0]
                if outputs[1]>self.confidence:
                    flag = True
                    label_res.append(f"1_{outputs[1]}")
                else:
                    flag = True
                    label_res.append(f"0_{outputs[0]}")
            else:
                image_pil = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                nonlinear = nn.Softmax(dim=1)
                sample = self.transforms(Image.fromarray(image_pil)).unsqueeze(0)
                outputs = nonlinear(self.model(sample)).tolist()[0]
                argmax_index = np.argmax(outputs)
                flag = True
                label_res.append(f"{argmax_index}_{outputs[argmax_index]}")


            if flag:
                # 保存检测结果
                img_base64 = image2base64(image)
                res["img"] = img_base64
                res["detection"] = label_res
            return json.dumps(res), 200
        except Exception as e:
            print(e)
            return json.dumps(str(e)), 400


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type_map', type=str, default='nest', help='Single or multiple choice.')
    # parser.add_argument('--yolo_version', type=str, default='yolov8', help='Detection model.')
    opt = parser.parse_args()
    return opt


def port_is_used(port, ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        print('%s:%d is used' % (ip, port))
        return True
    except:
        return False


if __name__ == '__main__':
    opt = vars(parse_opt())
    print(opt)
    base_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_path,"model_conf.cfg")) as f:
        model_cfg = json.load(f)

    # 部署模型
    # model_type = opt['model_type'].strip()
    model = opt['model_type_map'].strip()
    # for model_ in model_cfg:
    #     if model_type in model_cfg[model_]['model_type']:
    #         model = model_

    host_number = str(model_cfg[model]['port'])
    # 通过端口占用判断模型是否启动，如已启动，程序退出
    if port_is_used(int(host_number)):
        exit(0)

    print('开始部署 %s 模型'%model)
    model_path = model_cfg[model]['path']
    worker_threads = 2
    io_threads = 2
    worker_processes = model_cfg[model]['model_count']
    process = Processor(worker_threads=worker_threads, io_threads=io_threads, worker_processes=worker_processes,
                        host_number=host_number, model_path=model_path, model_type=model,
                        yolo_version=model_cfg[model]['yolo_version'],confidence=model_cfg[model]['confidence'])
    process.run()
