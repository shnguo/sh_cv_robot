import argparse
import orjson
import time
from log import get_logger
import redis
from image_utils import base642image,image2base64,image2base64_new,cv2AddChineseText
import cv2
from threading import Thread,local
import threading
import requests
import json
import orjson
import os
from datetime import datetime
import cv2
import numpy as np
import requests
import math
from PIL import Image
from clf_conf import clf_conf
import queue
from translations import translation_conf
logger = get_logger(os.path.basename(__file__))
base_path = os.path.dirname(os.path.abspath(__file__))
develop = os.getenv('GS_DEVELOP')

# develop=True

class Detection_Post(threading.Thread):
    def __init__(self,img_raw_cv,url):
        threading.Thread.__init__(self)
        # self.img_raw = cv2.cvtColor(img_raw_cv, cv2.COLOR_RGB2BGR)
        self.img_raw_b64 = image2base64(img_raw_cv)
        self.url = url

    def run(self):
        count = 0
        while True:
            try:
                r = requests.post(self.url,json={'img_raw':self.img_raw_b64},timeout=60)
                break
            except Exception as e:
                logger.error(e)
                count = count+1
                time.sleep(5)
                if count>100:
                        break
        self.result = orjson.loads(r.text)
    
    def get_result(self):
        return self.result

class Event_Sender(threading.Thread):

    def __init__(self,scene,event_time,result,result_queue,host,**kwargs):
        threading.Thread.__init__(self)
        self.scene=scene
        self.event_time = event_time
        self.timestamp = int(round(datetime.strptime(event_time,"%Y-%m-%d %H:%M:%S:%f").timestamp()))
        self.result = result
        self.dispatch = {
            'voltage_line_matter':self.common_solver,
            'oil':self.common_solver,
            'smoke':self.common_solver,
            'helmet':self.helmet,
            'suit':self.suit,
            'smoking':self.smoking,
            'insulator_broken':self.insulator_broken,
            'insulator_stain':self.insulator_stain,
            'capacitor_bulge':self.clf_clf_solver,
            'box_door':self.clf_clf_solver,
            'blurred_dial':self.clf_clf_solver,
            'abnormal_meter':self.clf_solver,
            'silicagel':self.clf_clf_solver,
            'screen_crash':self.clf_clf_solver,
            'damaged':self.clf_clf_solver,
            'person':self.common_solver,
            'oil_level':self.oil_level_solver,
            'switch_1':self.clf_clf_solver,
            'switch_2':self.clf_clf_solver,
            'surface_pollution':self.clf_solver,
            'rust':self.common_solver,
            'fire':self.common_solver,
            'light_status':self.clf_clf_solver,
            'yaban':self.clf_clf_solver,
            'open_door':self.clf_solver,
            'box_door_2':self.box_door_2_solver,

        }
        self.url = f"http://{host}:8090/api/v1/result"
        self.kwargs = kwargs
        self.result_queue = result_queue
    
    def run(self):
        self.dispatch[self.scene]()
    
    def post_report(self,img,result,model_name):
        image = ''
        if img.any() and result>-1:
            image=image2base64(img)
            if develop:
                cv2.imwrite(os.path.join(base_path,'./result/'+model_name+f'_{result}_'+f'{datetime.now()}.jpg'),img)
        data = {
            'date':self.timestamp,
            'image':image,
            'result':str(result),
             "history_info_id":self.kwargs['history_info_id'],
             "history_type":self.kwargs["history_type"],
             'model_type':self.kwargs["model_type"],
        }
        self.result_queue.put(data)
        if develop:
            print(data['result'])
        try:
            requests.post(self.url,json=data,timeout=3)
        except Exception as e:
            logger.error(e)


    
    def common_solver(self):
        post_result = -1
        for l in self.result:
            for _r in l:
                # print(f"{_r}:{l[_r]['detection']}")
                if l[_r]['img']:
                    post_result = 1
                    img = base642image(l[_r]['img'])
                    for _info in l[_r]["detection"]:
                        _info_list = _info.split('_')
                        cv2.rectangle(img, (int(float(_info_list[2])), int(float(_info_list[3]))), (int(float(_info_list[4])), int(float(_info_list[5]))),
                                    (0, 0, 255),thickness=2)
                        img = cv2AddChineseText(img,translation_conf.get(_r,""),(int(float(_info_list[2])),max(int(float(_info_list[3]))-18,0)),(255, 0, 0),15)
                        # cv2.putText(img,translation_conf.get(_r,""),(int(float(_info_list[2])),max(int(float(_info_list[3]))-10,0)),cv2.FONT_HERSHEY_COMPLEX, 5,(0, 0, 255),2)
                    self.post_report(img,post_result,_r) 
        if  post_result==-1:
            self.post_report(np.array([]),post_result,'')

    def clf_solver(self):
        for l in self.result:
            for _r in l:
                if develop:
                    print(f"{_r}:{l[_r]['detection']}")
                if l[_r]['img']:
                    post_result = int(float(l[_r]['detection'][0].split('_')[0]))
                    img = base642image(l[_r]['img'])
                    self.post_report(img,post_result,_r)
    
    def yolo_clf_solver(self):
        for l in self.result:
            if 'objects_10' in l:
                objects_10_value = l['objects_10']
                break
        find = False
        if objects_10_value['img']:
            print(objects_10_value['detection'])
            for _d in objects_10_value['detection']:
                if int(float(_d.split('_')[0]))==clf_conf[self.scene]:
                    find = True
                    img=base642image(objects_10_value['img'])
                    break
        if find:
            for l in self.result:
                if f'{self.scene}_clf' in l:
                    post_result = int(float(l[f'{self.scene}_clf']['detection'][0].split('_')[0]))
                    self.post_report(img,post_result,f'{self.scene}_clf')

        else:
            self.post_report(np.array([]),-1,'')
                
    def clf_clf_solver(self):
        for l in self.result:
            if 'objects_10_clf' in l:
                objects_10_value = l['objects_10_clf']
                break
        find = False
        # print(f"clf1={objects_10_value['detection'][0].split('_')}")
        img=base642image(objects_10_value['img'])
        if int(float(objects_10_value['detection'][0].split('_')[0]))==clf_conf[self.scene]:
            find=True
        if find:
            for l in self.result:
                if f'{self.scene}_clf' in l:
                    post_result = int(float(l[f'{self.scene}_clf']['detection'][0].split('_')[0]))
                    # print(f"bool={l[f'{self.scene}_clf']['detection'][0].split('_')}")
                    self.post_report(img,post_result,f'{self.scene}_clf')

        else:
            self.post_report(np.array([]),-1,'')

        
    
    def box_door(self):
        post_result = -1
        for l in self.result:
            for _r in l:
                # print(f"{_r}:{l[_r]['detection']}")
                if _r=='box_door':
                    l_box_door = l
                elif _r=='person':
                    l_person = l
        if l_box_door['box_door']['img'] and not l_person['person']['img']:
            post_result = 1
            img = base642image(l_box_door['box_door']['img'])
            self.post_report(img,post_result,'box_door')
        else:
            post_result = 0
            img = base642image(l_box_door['box_door']['img'])
            self.post_report(img,post_result,'box_door')
        if  post_result==-1:
            self.post_report(np.array([]),post_result,'box_door')
            
    
    def helmet(self):
        post_result = -1
        target_dict = self.result[0]['helmet']
        if target_dict['img']:
            img = base642image(target_dict['img'])
            print(target_dict['detection'])
            for _item in target_dict['detection']:
                _info_list = _item.split('_')
                if int(float(_info_list[0]))>0:
                    cv2.rectangle(img, (int(_info_list[2]), int(_info_list[3])), (int(_info_list[4]), int(_info_list[5])),
                                    (0, 0, 255),thickness=2)
                    img = cv2AddChineseText(img,"头盔异常",(int(float(_info_list[2])),max(int(float(_info_list[3]))-18,0)),(255, 0, 0),15)
                    post_result=1
                else:
                    cv2.rectangle(img, (int(_info_list[2]), int(_info_list[3])), (int(_info_list[4]), int(_info_list[5])),
                                    (0, 255, 0),thickness=2)
                    img = cv2AddChineseText(img,"头盔正常",(int(float(_info_list[2])),max(int(float(_info_list[3]))-18,0)),(0, 255, 0),15)
                    if post_result<0:
                        post_result=0
            self.post_report(img,post_result,'helmet')
        else:
            self.post_report(np.array([]),post_result,'helmet')
    
    def suit(self):
        post_result = -1
        target_dict = self.result[0]['helmet_suit_smoking']
        if target_dict['img']:
            sign, img = self.helmet_suit_smoking(target_dict['detection'],target='suit',img = base642image(target_dict['img']))
            if sign:
                post_result=1
                self.post_report(img,post_result,'suit')
            else:
                self.post_report(img,0,'suit') 
        else:
            self.post_report(np.array([]),post_result,'suit')

    def smoking(self):
        post_result=-1
        target_dict = self.result[0]['helmet_suit_smoking']
        if target_dict['img']:
            sign, img = self.helmet_suit_smoking(target_dict['detection'],target='smoking',img = base642image(target_dict['img']))
            if sign:
                post_result=1
                self.post_report(img,post_result,'smoking') 
            else:
                self.post_report(img,0,'smoking')
        else:
            self.post_report(np.array([]),post_result,'smoking')
    
    def insulator_broken(self):
        post_result=-1
        target_dict = self.result[0]['insulator']
        if target_dict['img']:
            sign, img = self.insulator(target_dict['detection'],target='broken',img = base642image(target_dict['img']))
            if sign:
                post_result=1
                self.post_report(img,post_result,'insulator_broken') 
            else:
                self.post_report(img,0,'insulator_broken')
        else:
            self.post_report(np.array([]),post_result,'insulator_broken')

    def insulator_stain(self):
        post_result=-1
        target_dict = self.result[0]['insulator']
        if target_dict['img']:
            sign, img = self.insulator(target_dict['detection'],target='stain',img = base642image(target_dict['img']))
            if sign:
                post_result=1
                self.post_report(img,post_result,'insulator_stain') 
            else:
                self.post_report(img,0,'insulator_stain')
        else:
            self.post_report(np.array([]),post_result,'insulator_stain')

    def helmet_suit_smoking(self, ls, target, img, confidence_level=0.5):
        ls = np.array([s.split('_') for s in ls]).astype(float)
        ls = ls[ls[:, 1] > confidence_level, :]
        unique, counts = np.unique(ls[:, 0], return_counts=True)
        
        if target == 'helmet':
            sign = not((1.0 in unique) and (0.0 in unique) and (counts[0] >= counts[1])) 
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 1), :]: # plot rectangle for person.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (255, 0, 0),
                    2
                )
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 0), :]: # plot rectangle for helmet.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (225, 225, 0),
                    2
                )
            return sign, img
        
        if target == 'suit':
            sign = ((1 in unique) and (2 in unique)) or ((1 in unique) and (3 in unique))
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 1), :]: # plot rectangle for person.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (0, 255, 0),
                    2
                )
                img = cv2AddChineseText(img,"着装正常",(int(item[2]),max(int(item[3])-18,0)),(0, 255, 0),15)
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 2), :]: # plot rectangle for short sleeve.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 255),
                    2
                )
                img = cv2AddChineseText(img,"着装异常",(int(item[2]),max(int(item[3])-18,0)),(255, 0, 0),15)
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 3), :]: # plot rectangle for short pants.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 225),
                    2
                )
                img = cv2AddChineseText(img,"着装异常",(int(item[2]),max(int(item[3])-18,0)),(255, 0, 0),15)
            return sign, img
        
        if target == 'smoking':
            sign = (1 in unique) and (4 in unique)
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 1), :]: # plot rectangle for person.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (0, 255, 0),
                    2
                )
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 4), :]: # plot rectangle for cigarette.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (0, 0, 255),
                    2
                )
                img = cv2AddChineseText(img,"吸烟异常",(int(item[2]),max(int(item[3])-18,0)),(255, 0, 0),15)
            return sign, img   
    
    def insulator(self, ls, target, img, confidence_level=0.5):
        ls = np.array([s.split('_') for s in ls]).astype(float)
        ls = ls[ls[:, 1] > confidence_level, :]
        unique, counts = np.unique(ls[:, 0], return_counts=True)
        
        if target == 'broken':
            sign = (3 in unique) and (0 in unique)
            for item in ls[np.logical_and(ls[:, 1] > 0.5, ls[:, 0] == 3), :]: # plot rectangle for insulator.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (0, 255, 0),
                    2
                )
            for item in ls[np.logical_and(ls[:, 1] > 0.5, ls[:, 0] == 0), :]: # plot rectangle for broken part.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 255),
                    2
                )
                img = cv2AddChineseText(img,"绝缘子破损",(int(item[2]),max(int(item[3])-18,0)),(255, 0, 0),15)
            return sign, img
        
        if target == 'stain':
            sign = (3 in unique) and (1 in unique)
            for item in ls[np.logical_and(ls[:, 1] > 0.5, ls[:, 0] == 3), :]: # plot rectangle for insulator.
                img = cv2.rectangle(
                    img, 
                    (int(item[2]), int(item[3])), 
                    (int(item[4]), int(item[5])),
                    (0, 255, 0),
                    2
                ) 
            for item in ls[np.logical_and(ls[:, 1] > 0.5, ls[:, 0] == 1), :]: # plot rectangle for stain part.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 255),
                    2
                )
                img = cv2AddChineseText(img,"绝缘子污迹",(int(item[2]),max(int(item[3])-18,0)),(255, 0, 0),15)
            return sign, img
    
    def oil_reading(self,bound1, bound2, center, tail):
        b1 = np.arcsin((center[1] - bound1[1]) / np.sqrt((center[0] - bound1[0]) ** 2 + (center[1] - bound1[1]) ** 2))
        if center[0] > bound1[0]:
            b1 = math.pi - b1
        if b1 < 0:
            b1 += 2 * math.pi

        b2 = np.arcsin((center[1] - bound2[1]) / np.sqrt((center[0] - bound2[0]) ** 2 + (center[1] - bound2[1]) ** 2))
        if center[0] > bound2[0]:
            b2 = math.pi - b2
        if b2 < 0:
            b2 += 2 * math.pi

        r = np.arcsin((center[1] - tail[1]) / np.sqrt((center[0] - tail[0]) ** 2 + (center[1] - tail[1]) ** 2))
        r = -r
        if center[0] < tail[0]:
            r = math.pi - r
        if r < 0:
            r += 2 * math.pi

        if b1 > b2:
            return (b1 - r) / (b1 - b2)
        return (b2 - r) / (b2 - b1)

    def oil_level(self,ls, img, confidence_level=0.1):
        """
        ls: list of string, eg: ['1.0_0.56_968_204_1053_276'] xywh
        return values: float (percentage), img
        If no enough points are detected, value will be -1
        ["bound1", "bound2", "center", "tail"]
        """
        ls = np.array([s.split('_') for s in ls]).astype(float)
        ls = ls[ls[:, 1] > confidence_level, :]
        unique, counts = np.unique(ls[:, 0], return_counts=True)
        points = []
        for item in unique:
            tmp = ls[ls[:, 0] == item, 2:].mean(axis=0)
            points.append([(tmp[0]+tmp[2])/2, (tmp[1]+tmp[3])/2])
            img = cv2.circle(
                img, 
                (int(points[int(item)][0]), int(points[int(item)][1])), 
                radius=5, 
                color=(0, 255, 255), 
                thickness=-1
                            )
        if len(unique) < 4:
            return -1, img
        
        else:
            return self.oil_reading(points[0], points[1], points[2], points[3]), img

    def oil_level_solver(self):
        post_result=-1
        target_dict = self.result[0]['oil_level']
        # print(target_dict['detection'])
        if target_dict['img']:
            result, img = self.oil_level(target_dict['detection'],img = base642image(target_dict['img']))
            self.post_report(img,result,'oil_level') 
        else:
            self.post_report(np.array([]),-1,'oil_level')
    
    def box_door_2_solver(self):
        post_result=-1
        target_dict = self.result[0]['door_yolo']
        if target_dict['img']:
            img = base642image(target_dict['img'])
            for _item in target_dict['detection']:
                _info_list = _item.split('_')
                if int(float(_info_list[0]))>0:
                    cv2.rectangle(img, (int(float(_info_list[2])), int(float(_info_list[3]))), (int(float(_info_list[4])), int(float(_info_list[5]))),
                                    (0, 0, 255),thickness=2)
                    img = cv2AddChineseText(img,"关门异常",(int(float(_info_list[2])),max(int(float(_info_list[3]))-18,0)),(0, 255, 0),15)
                    post_result=1
                else:
                    cv2.rectangle(img, (int(float(_info_list[2])), int(float(_info_list[3]))), (int(float(_info_list[4])), int(float(_info_list[5]))),
                                    (0, 255, 0),thickness=2)
                    img = cv2AddChineseText(img,"关门正常",(int(float(_info_list[2])),max(int(float(_info_list[3]))-18,0)),(0, 255, 0),15)
                    if post_result<0:
                        post_result=0
            self.post_report(img,post_result,'door_yolo')

        else:
            self.post_report(np.array([]),-1,'door_yolo')




    # def light_status(self, ls, target, img, confidence_level=0.5):
    #     ls = np.array([s.split('_') for s in ls]).astype(float)
    #     ls = ls[ls[:, 1] > confidence_level, :]
    #     unique, counts = np.unique(ls[:, 0], return_counts=True)
        
    #     if target == 'close':
    #         sign = 1 in unique
    #         for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 1), :]: 
    #             img = cv2.rectangle(
    #                 img, 
    #                 (int(item[2]), int(item[3])), 
    #                 (int(item[4]), int(item[5])),
    #                 (0, 255, 0),
    #                 2
    #             )
                
    #         return sign, img
        
    #     if target == 'open':
    #         sign = 0 in unique
    #         for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 0), :]: 
    #             img = cv2.rectangle(
    #                 img, 
    #                 (int(item[2]), int(item[3])), 
    #                 (int(item[4]), int(item[5])),
    #                 (0, 255, 0),
    #                 2
    #             ) 

    #         return sign, img


    # def light_status_solver(self):
    #     post_result = -1
    #     target_dict = self.result[0]['light_status']
    #     if target_dict['img']:
    #         post_result = 1
    #         sign, img = self.light_status(target_dict['detection'],'open',img = base642image(target_dict['img']))
    #         if sign:
    #             self.post_report(img,1,'light_status')
    #         else:
    #             self.post_report(img,0,'light_status')

    #     else:
    #         self.post_report(np.array([]),post_result,'light_status')

    # def yaban(self, ls, target, img, confidence_level=0.5):
    #     """
    #     ls: list of string, eg: ['1.0_0.56_968_204_1053_276']
    #     target: one of the targets ['close', 'open']
    #     return values: logical value (True / False), img
    #     """
    #     ls = np.array([s.split('_') for s in ls]).astype(float)
    #     ls = ls[ls[:, 1] > confidence_level, :]
    #     unique, counts = np.unique(ls[:, 0], return_counts=True)
        
    #     if target == 'close':
    #         sign = 2 in unique
    #         for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 2), :]: 
    #             img = cv2.rectangle(
    #                 img, 
    #                 (int(item[2]), int(item[3])), 
    #                 (int(item[4]), int(item[5])),
    #                 (0, 255, 0),
    #                 2
    #             )
                
    #         return sign, img
        
    #     if target == 'open':
    #         sign = 3 in unique
    #         for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 3), :]: 
    #             img = cv2.rectangle(
    #                 img, 
    #                 (int(item[2]), int(item[3])), 
    #                 (int(item[4]), int(item[5])),
    #                 (0, 255, 0),
    #                 2
    #             ) 

    #         return sign, img
        
    # def yaban_solver(self):
    #     post_result = -1
    #     target_dict = self.result[0]['yaban']
    #     if target_dict['img']:
    #         post_result = 1
    #         sign, img = self.yaban(target_dict['detection'],'open',img = base642image(target_dict['img']))
    #         if sign:
    #             self.post_report(img,1,'yaban')
    #         else:
    #             self.post_report(img,0,'yaban')

    #     else:
    #         self.post_report(np.array([]),post_result,'yaban')

      

class SendPost(object):

    def __init__(self,source_url,forever,scene,model_list,model_conf,pic_area,docker=False,**kwargs):
        if docker:
            self.host = 'host.docker.internal'
        else:
            self.host = '0.0.0.0'
        self.detection_list = [f'http://{self.host}:'+str(model_conf[m]['port'])+'/test' for m in model_list]
        self.redis_con = redis.Redis(host=self.host,
                        password='foster123456',
                        decode_responses=True,socket_timeout=30)
        self.source_url = source_url
        self.model_list = model_list
        self.scene = scene      
        self.forever = forever
        self.kwargs = kwargs
        [self.x,self.y,self.w,self.h] = [int(x) for x in pic_area.split(',')]
        self.result_queue = queue.Queue()

    def read_rtsp_frame(self,key):
        return orjson.loads(self.redis_con.get(key))
         
    def run(self):
        if self.source_url.startswith('rtsp'):
            if self.forever:
                self.run_rtsp_gpu()
            else:
                self.run_rtsp_cpu()
        elif self.source_url.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', 
                                               '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            self.run_pic(self.source_url)
        
        elif self.source_url.lower().endswith('/'):
            for filename in os.listdir(self.source_url):
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', 
                                               '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    self.run_pic(os.path.join(self.source_url,filename))
        elif self.source_url.lower().endswith('mp4'):
            self.run_mp4_cpu()
        else: 
            return 0
    
    def run_mp4_cpu(self):
        cap = cv2.VideoCapture(self.source_url)
        while(cap.isOpened()):
            ret, img = cap.read()
            img  = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            self.detect_and_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"),img)



    def run_pic(self,file_path):
        img = cv2.imread(file_path)
        # I = Image.open(file_path)
        # img = np.array(I)
        self.detect_and_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"),img)
    
    def run_rtsp_cpu(self):
        cap = cv2.VideoCapture(self.source_url)
        ret,img = cap.read()
        # img  = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        self.detect_and_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"),img)

            
    def run_rtsp_gpu(self):
        ip = self.source_url.split('@')[-1]
        key = 'frame_'+ip
        read_fail_time = 0
        read_update_fail_time = 0
        pre_read_time = None
        while True:
            try:
                frame_info_dict = self.read_rtsp_frame(key)
            except Exception as e:
                logger.error(e)
                if read_fail_time>300:
                    return 0
                else:
                    time.sleep(5)
                    read_fail_time = read_fail_time+1
                    continue
            read_fail_time = 0

            if not frame_info_dict or frame_info_dict['now_time']==pre_read_time:
                logger.error(f'frame_{ip} is not update')
                if read_update_fail_time>300:
                    return 0
                else:
                    read_update_fail_time = read_update_fail_time+1
                    time.sleep(5)
                    continue
            read_update_fail_time = 0
            pre_read_time = frame_info_dict['now_time']
            img_raw_cv = base642image(frame_info_dict['img'])
            img_raw_cv  = cv2.cvtColor(img_raw_cv,cv2.COLOR_RGB2BGR)
            self.detect_and_send(frame_info_dict['now_time'],img_raw_cv)


    def detect_and_send(self,pic_time,img_raw_cv):               
        thread_pool = []
        if self.h>0 and self.w>0:
            img_raw_cv_h,img_raw_cv_w = img_raw_cv.shape[0],img_raw_cv.shape[1]
            # crop_y = int(self.y/225*img_raw_cv_h)
            # crop_h = int(self.h/225*img_raw_cv_h)
            # crop_x = int(self.x/400*img_raw_cv_w)
            # crop_w = int(self.w/400*img_raw_cv_w)

            crop_y = int(self.y)
            crop_h = int(self.h)
            crop_x = int(self.x)
            crop_w = int(self.w)

            crop = img_raw_cv[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        else:
            crop = img_raw_cv
        for url in self.detection_list:
            thread_pool.append(Detection_Post(crop,url))
        for t in thread_pool:
            t.start()
            t.join()
        result = []
        for m,t in zip(self.model_list,thread_pool):
            result.append({m:t.get_result()})
        th = Event_Sender(self.scene,pic_time,result,self.result_queue,self.host,**self.kwargs)
        th.start()
        th.join()

    def get_result(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        return {
            'results':results
        }
        
                




def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_type', type=str, default='rtsp', help='detection type')
    parser.add_argument('--source_url',type=str,  help='source url')
    parser.add_argument('--forever', action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--scene',type=str,default='voltage_line_matter',help='scene')
    parser.add_argument('--history_info_id',type=int,default=0,help='history_info_id')
    parser.add_argument('--history_type',type=int,default=0,help='history_type')
    parser.add_argument('--model_type',type=int,default=0,help='model_type')
    parser.add_argument('--pic_area',type=str,default='0,0,0,0',help='x,y,w,h')
    opt = parser.parse_args()
    return opt
 

def main():
    opt = vars(parse_opt())
    logger.info('开始检测')
    if develop:
        print(opt)
    # time.sleep(30)
    base_path = os.path.dirname(os.path.abspath(__file__))
    scene = opt['scene'].strip()
    with open(os.path.join(base_path,'scene_map.cfg')) as f:
        scene_map = json.load(f)
    model_list = scene_map.get(scene,None)
    if not model_list:
        logger.error('scene is not correct')
        return 0     
    with open(os.path.join(base_path,"model_conf.cfg")) as f:
        model_conf = json.load(f)
    sp = SendPost(opt['source_url'],opt['forever'],scene,model_list,model_conf,pic_area=opt['pic_area'],
                  history_info_id=opt['history_info_id'],history_type=opt['history_type'],model_type=opt['model_type'])
    sp.run()
    



if __name__=='__main__':
    main()