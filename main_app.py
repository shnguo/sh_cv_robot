# from datetime import datetime, timedelta
# from os import lseek
# from typing import Optional
from fastapi import FastAPI

# import pandas as pd
import uvicorn
from pydantic import BaseModel
from typing import Dict, Union, List, Literal
from fastapi.middleware.cors import CORSMiddleware
import os
import log

# from fastapi.responses import JSONResponse
import time
import traceback
from fastapi import Request, Response
import json

# from image_utils import base642image
# import io
# import socket
# from passlib.hash import md5_crypt
# from redis import asyncio as aioredis
# from read_video import get_frame,get_label_img
from fastapi.responses import ORJSONResponse

# from model_type_map import model_type_map
from send_post import SendPost
import cv2
from starlette.responses import StreamingResponse
import io
from yiwu import process_video,process_video_gate

logger = log.get_logger(os.path.basename(__file__))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Params(BaseModel):
    source_url: str
    scene: str
    pic_area: str = "0,0,0,0"
    history_info_id: int = 0
    history_type: int = 0
    model_type: int = 0


class Rtsp(BaseModel):
    source_url: str

class Video_Params(BaseModel):
    source_url: str
    gate:bool=False
    history_info_id: int = 0
    history_type: int = 0
    model_type: int = 0

@app.on_event("startup")
async def _startup():
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, "scene_map.cfg")) as f:
        app.state.scene_map = json.load(f)
    with open(os.path.join(base_path, "model_conf.cfg")) as f:
        app.state.model_conf = json.load(f)
    logger.debug("startup done")


@app.on_event("shutdown")
async def shutdown_event():
    logger.debug("shutdown done")


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    start = time.time()
    response = Response("Internal server error", status_code=500)
    try:
        response = await call_next(request)

    except Exception:
        error_data = [
            f"status: {response.status_code}\n",
            f"params: {request.query_params}\n",
            f"path_params: {request.url.path}\n",
            f"time: {time.time() - start}\n",
            f"traceback: {traceback.format_exc()[-2000:]}",
        ]

        error_msg = "".join(error_data)
        logger.error(error_msg)

    end = time.time()
    logger.info(
        f"{request.client.host}:{request.client.port} {request.url.path} {response.status_code} took {round(end-start,5)}"
    )
    return response


# 创建访问路径
@app.get("/")
# @cache(expire=60)
async def read_root():  # 定义根目录方法
    return ORJSONResponse({"result": "Hello World"})


@app.post("/sendpost")
async def sendpost(pa: Params):
    model_list = app.state.scene_map.get(pa.scene, None)
    if not model_list:
        logger.error("scene is not correct")
        return {"error": "scene is not correct"}
    sp = SendPost(
        pa.source_url,
        False,
        pa.scene,
        model_list,
        app.state.model_conf,
        pic_area=pa.pic_area,
        docker=True,
        history_info_id=pa.history_info_id,
        history_type=pa.history_type,
        model_type=pa.model_type,
    )
    sp.run()
    return sp.get_result()


def get_frame(source_url):
    ip = source_url
    cap = cv2.VideoCapture(ip)
    if not cap.isOpened():
        return None
    fps = int(cv2.CAP_PROP_FPS) if int(cv2.CAP_PROP_FPS) >= 1000 else 25
    i = 0
    while True:
        retval, frame = cap.read()
        if i == fps:
            return frame
        i += 1


@app.post("/curpic_get")
async def curpic_get(rtsp: Rtsp):
    frame = get_frame(rtsp.source_url)
    pic = cv2.resize(frame, (500, int(500 * frame.shape[0] / frame.shape[1])))
    print(f"pic.shape={pic.shape}")
    res, im_png = cv2.imencode(".png", pic)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/video_yiwu")
async def video_yiwu(vp:Video_Params):
    data = {
        'result':-1,
        'image':'',
        "history_info_id":vp.history_info_id,
        "history_type":vp.history_type,
        'model_type':vp.model_type,
    }
    if vp.gate:
        data["result"],data['image']=process_video_gate(vp.source_url)
    else:
        data["result"],data['image']=process_video(vp.source_url)
    return data



if __name__ == "__main__":
    uvicorn.run(app="main_app:app", host="0.0.0.0", port=8002, reload=True)
