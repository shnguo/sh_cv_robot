import argparse
import orjson
import time
from log import get_logger
import redis
from image_utils import base642image,image2base64
import cv2
from threading import Thread,local
import threading
import requests
import json
import orjson
import os
from datetime import datetime
import cv2
logger = get_logger(__file__) 


class Detection_Post(threading.Thread):
    def __init__(self,img_raw_cv,url):
        threading.Thread.__init__(self)
        self.img_raw = cv2.cvtColor(img_raw_cv, cv2.COLOR_RGB2BGR)
        self.img_raw_b64 = image2base64(self.img_raw)
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

    def __init__(self,scene,event_time,result):
        threading.Thread.__init__(self)
        self.scene=scene
        self.event_time = event_time
        self.result = result
        self.dispatch = {
            'voltage_line_matter':self.voltage_line_matter
        }
    
    def run(self):
        self.dispatch[self.scene]()
    
    def voltage_line_matter(self):
        print(self.event_time)
        for l in self.result:
            for _r in l:
                print(f"{_r}:{l[_r]['detection']}")
                if l[_r]['img']:
                    cv2.imwrite('./result/'+_r+'.jpg',base642image(l[_r]['img']))
        

class SendPost(object):

    def __init__(self,source_url,forever,scene,model_list,model_conf):
        self.detection_list = ['http://0.0.0.0:'+str(model_conf[m]['port'])+'/test' for m in model_list]
        self.source_url = source_url
        self.model_list = model_list
        self.scene = scene
        self.redis_con = redis.Redis(host='0.0.0.0',
                        password='foster123456',
                        decode_responses=True,socket_timeout=30)
        self.forever = forever

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
                

    def run_pic(self,file_path):
        img = cv2.imread(file_path)
        self.detect_and_send(datetime.now(),img)
    
    def run_rtsp_cpu(self):
        cap = cv2.VideoCapture(self.source_url)
        ret,img = cap.read()
        self.detect_and_send(datetime.now(),img)

            
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
            self.detect_and_send(frame_info_dict['now_time'],img_raw_cv)


    def detect_and_send(self,pic_time,img_raw_cv):               
        thread_pool = []
        for url in self.detection_list:
            thread_pool.append(Detection_Post(img_raw_cv,url))
        for t in thread_pool:
            t.start()
            t.join()
        result = []
        for m,t in zip(self.model_list,thread_pool):
            result.append({m:t.get_result()})
        th = Event_Sender(self.scene,pic_time,result)
        th.start()
                




def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--source_type', type=str, default='rtsp', help='detection type')
    parser.add_argument('--source_url',type=str,  help='source url')
    parser.add_argument('--forever', action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--scene',type=str,default='voltage_line_matter',help='scene')
    opt = parser.parse_args()
    return opt
 

def main():
    opt = vars(parse_opt())
    logger.info('开始检测')
    print(opt)
    # time.sleep(30)
    scene = opt['scene'].strip()
    with open('scene_map.cfg') as f:
        scene_map = json.load(f)
    model_list = scene_map.get(scene,None)
    if not model_list:
        logger.error('scene is not correct')
        return 0     
    with open("model_conf.cfg") as f:
        model_conf = json.load(f)
    sp = SendPost(opt['source_url'],opt['forever'],scene,model_list,model_conf)
    sp.run()
    



if __name__=='__main__':
    main()