import argparse
import orjson
import time
from log import get_logger
import redis
from image_utils import base642image,image2base64,image2base64_new
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
logger = get_logger(os.path.basename(__file__))
base_path = os.path.dirname(os.path.abspath(__file__))

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

    def __init__(self,scene,event_time,result,**kwargs):
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
            'capacitor_bulge':self.clf_solver,
            'box_door':self.box_door,
            'blurred_dial':self.clf_solver,
            'abnormal_meter':self.clf_solver,
            'silicagel':self.clf_solver,
            'screen_crash':self.clf_solver,
            'damaged':self.clf_solver,
            'person':self.common_solver,
            'oil_level':self.oil_level_solver
        }
        self.url = "http://localhost:8090/api/v1/result"
        self.kwargs = kwargs
    
    def run(self):
        self.dispatch[self.scene]()
    
    def post_report(self,img,result,model_name):
        image = ''
        if img.any() and result:
            image=image2base64(img)
            cv2.imwrite(os.path.join(base_path,'./result/'+model_name+f'{datetime.now()}.jpg'),img)
        data = {
            'date':self.timestamp,
            'image':image,
            'result':result,
             "history_info_id":self.kwargs['history_info_id'],
             "history_type":self.kwargs["history_type"]
        }
        # print(data)
        try:
            requests.post(self.url,json=data,timeout=3)
        except Exception as e:
            logger.error(e)


    
    def common_solver(self):
        post_result = 0
        for l in self.result:
            for _r in l:
                print(f"{_r}:{l[_r]['detection']}")
                if l[_r]['img']:
                    post_result = 1
                    img = base642image(l[_r]['img'])
                    for _info in l[_r]["detection"]:
                        _info_list = _info.split('_')
                        cv2.rectangle(img, (int(_info_list[2]), int(_info_list[3])), (int(_info_list[4]), int(_info_list[5])),
                                    (0, 0, 255),thickness=2)
                    self.post_report(img,post_result,_r) 
        if  post_result==0:
            self.post_report(np.array([]),post_result,'')

    def clf_solver(self):
        post_result = 0
        for l in self.result:
            for _r in l:
                print(f"{_r}:{l[_r]['detection']}")
                if l[_r]['img']:
                    post_result = 1
                    img = base642image(l[_r]['img'])
                    self.post_report(img,post_result,_r)
        if  post_result==0:
            self.post_report(np.array([]),post_result,'')
    
    def box_door(self):
        post_result = 0
        for l in self.result:
            for _r in l:
                print(f"{_r}:{l[_r]['detection']}")
                if _r=='box_door':
                    l_box_door = l
                elif _r=='person':
                    l_person = l
        if l_box_door['box_door']['img'] and not l_person['person']['img']:
            post_result = 1
            img = base642image(l_box_door['box_door']['img'])
            self.post_report(img,post_result,'box_door')
        if  post_result==0:
            self.post_report(np.array([]),post_result,'')
            
    
    def helmet(self):
        post_result = 0
        target_dict = self.result[0]['helmet_suit_smoking']
        if target_dict['img']:
            # print(target_dict['detection'])
            sign, img = self.helmet_suit_smoking(target_dict['detection'],target='helmet',img = base642image(target_dict['img']))
            if sign:
                post_result=1
            self.post_report(img,post_result,'helmet') 
        else:
            self.post_report(np.array([]),post_result,'')
    
    def suit(self):
        post_result = 0
        target_dict = self.result[0]['helmet_suit_smoking']
        if target_dict['img']:
            sign, img = self.helmet_suit_smoking(target_dict['detection'],target='suit',img = base642image(target_dict['img']))
            if sign:
                post_result=1
            self.post_report(img,post_result,'suit') 
        else:
            self.post_report(np.array([]),post_result,'')

    def smoking(self):
        post_result=0
        target_dict = self.result[0]['helmet_suit_smoking']
        if target_dict['img']:
            sign, img = self.helmet_suit_smoking(target_dict['detection'],target='smoking',img = base642image(target_dict['img']))
            if sign:
                post_result=1
            self.post_report(img,post_result,'suit') 
        else:
            self.post_report(np.array([]),post_result,'')
    
    def insulator_broken(self):
        post_result=0
        target_dict = self.result[0]['insulator']
        if target_dict['img']:
            sign, img = self.insulator(target_dict['detection'],target='broken',img = base642image(target_dict['img']))
            if sign:
                post_result=1
            self.post_report(img,post_result,'insulator_broken') 
        else:
            self.post_report(np.array([]),post_result,'')

    def insulator_stain(self):
        post_result=0
        target_dict = self.result[0]['insulator']
        if target_dict['img']:
            sign, img = self.insulator(target_dict['detection'],target='stain',img = base642image(target_dict['img']))
            if sign:
                post_result=1
            self.post_report(img,post_result,'insulator_stain') 
        else:
            self.post_report(np.array([]),post_result,'')

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
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 2), :]: # plot rectangle for short sleeve.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 255),
                    2
                )
            for item in ls[np.logical_and(ls[:, 1] > confidence_level, ls[:, 0] == 3), :]: # plot rectangle for short pants.
                img = cv2.rectangle(
                    img,
                    (int(item[2]), int(item[3])),
                    (int(item[4]), int(item[5])),
                    (0, 0, 225),
                    2
                )
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
                    (255, 0, 255),
                    2
                )
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

    def oil_level(self,ls, img, confidence_level=0.5):
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
        post_result=0
        target_dict = self.result[0]['oil_level']
        if target_dict['img']:
            result, img = self.insulator(target_dict['detection'],img = base642image(target_dict['img']))
            self.post_report(img,result,'oil_level') 

      

class SendPost(object):

    def __init__(self,source_url,forever,scene,model_list,model_conf,pic_area,**kwargs):
        self.detection_list = ['http://0.0.0.0:'+str(model_conf[m]['port'])+'/test' for m in model_list]
        self.source_url = source_url
        self.model_list = model_list
        self.scene = scene
        self.redis_con = redis.Redis(host='0.0.0.0',
                        password='foster123456',
                        decode_responses=True,socket_timeout=30)
        self.forever = forever
        self.kwargs = kwargs
        [self.x,self.y,self.w,self.h] = [int(x) for x in pic_area.split(',')]

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
            crop = img_raw_cv[self.y:self.y+self.h, self.x:self.x+self.w]
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
        th = Event_Sender(self.scene,pic_time,result,**self.kwargs)
        th.start()
        th.join()
        
                




def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_type', type=str, default='rtsp', help='detection type')
    parser.add_argument('--source_url',type=str,  help='source url')
    parser.add_argument('--forever', action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--scene',type=str,default='voltage_line_matter',help='scene')
    parser.add_argument('--history_info_id',type=int,default=0,help='history_info_id')
    parser.add_argument('--history_type',type=int,default=0,help='history_type')
    parser.add_argument('--pic_area',type=str,default='0,0,0,0',help='x,y,w,h')
    opt = parser.parse_args()
    return opt
 

def main():
    opt = vars(parse_opt())
    logger.info('开始检测')
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
    sp = SendPost(opt['source_url'],opt['forever'],scene,model_list,model_conf,pic_area=opt['pic_area'],history_info_id=opt['history_info_id'],history_type=opt['history_type'])
    sp.run()
    



if __name__=='__main__':
    main()