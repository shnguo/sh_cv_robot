import json
import subprocess
from io import BytesIO
import cv2
from datetime import datetime
import requests
from image_utils import image2base64
import PyNvCodec as nvc
import time
# from memcached_client import MemcachedClient
import argparse
import numpy as np
import redis
import orjson

class ReadVideo(object):
    def __init__(self, ip, ports=None, gpu_id=None, web_ip=None):
        self.cap = cv2.VideoCapture(ip)
        self.ip = str(ip)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if int(self.fps) == 90000:
            self.fps = 25
        # mc = MemcachedClient(ports)
        # self.client = mc.run()

        self.gpu_id = gpu_id
        self.web_ip = web_ip
        self.redis_con = redis.Redis(host='0.0.0.0',
                        password='foster123456',
                        decode_responses=True,socket_timeout=30)

    def read(self):
        print('Reading video by CPU.')
        if '.mp4' in self.ip:
            try:
                time.sleep(5)
                while self.cap.isOpened():
                    fps_div_three = int(self.fps / 3)
                    one_frame_cnt = 0
                    for i in range(int(self.fps)):
                        self.cap.grab()
                        if i % fps_div_three == 0:
                            _, img = self.cap.retrieve()
                            one_frame_cnt += 1

                            if _:
                                now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                key = '3_frame_' + self.ip
                                value = {'img': image2base64(img), 'now_time': now_time}
                                try:
                                    self.client.set(key, value)
                                    # print("set:", "key:", key, "now_time:", now_time)
                                    if one_frame_cnt == 3:
                                        key = '1_frame_' + self.ip
                                        value = {'img': image2base64(img), 'now_time': now_time}
                                        self.client.set(key, value)
                                        print("set:", "key:", key, "now_time:", now_time)
                                except Exception as mem_e:
                                    res = requests.post('http://' + self.web_ip + '/strategy_msg',
                                                        json={'ip': ip.split('@')[-1], 'valid': False,
                                                              'model': '',
                                                              'msg': 'Memcached Set Error:' + str(mem_e) + '!',
                                                              'type': 'read_rtsp'})
                                    print("Memcached Set Error:", mem_e)
                                    return
            except Exception as e:
                print("Video Error:", e)
        else:
            try:
                while True:
                    fps_div_three = int(self.fps / 3)
                    fps_div_five = int(self.fps / 5)
                    for i in range(int(self.fps)):
                        self.cap.grab()
                        if i % fps_div_three == 0:
                            _, img = self.cap.retrieve()
                            if _:
                                now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                key = '3_frame_' + self.ip
                                value = {'img': image2base64(img), 'now_time': now_time}
                                try:
                                    self.client.set(key, value)
                                    print("set:", "key:", key, "now_time:", now_time)
                                except Exception as mem_e:
                                    res = requests.post('http://' + self.web_ip + '/strategy_msg',
                                                        json={'ip': ip.split('@')[-1], 'valid': False,
                                                              'model': '',
                                                              'msg': 'Memcached Set Error:' + str(mem_e) + '!',
                                                              'type': 'read_rtsp'})
                                    print("Memcached Set Error:", mem_e)
                                    return

                        if i % fps_div_five == 0:
                            _, img = self.cap.retrieve()
                            if _:
                                now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                key = '5_frame_' + self.ip
                                value = {'img': image2base64(img), 'now_time': now_time}
                                try:
                                    self.client.set(key, value)
                                    print("set:", "key:", key, "now_time:", now_time)
                                except Exception as mem_e:
                                    res = requests.post('http://' + self.web_ip + '/strategy_msg',
                                                        json={'ip': ip.split('@')[-1], 'valid': False,
                                                              'model': '',
                                                              'msg': 'Memcached Set Error:' + str(mem_e) + '!',
                                                              'type': 'read_rtsp'})
                                    print("Memcached Set Error:", mem_e)
                                    return
            except Exception as e:
                print("Video Error:", e)

    def read_by_gpu(self, length_seconds):
        # pass
        print('Reading video by GPU.')
        params = get_stream_params(self.ip)

        if not len(params):
            raise ValueError("Can not get " + self.ip + " streams params")

        w = params["width"]
        h = params["height"]
        f = params["format"]
        c = params["codec"]
        g = gpu_id

        # Prepare ffmpeg arguments
        if nvc.CudaVideoCodec.H264 == c:
            codec_name = "h264"
        elif nvc.CudaVideoCodec.HEVC == c:
            codec_name = "hevc"
        bsf_name = codec_name + "_mp4toannexb,dump_extra=all"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            self.ip,
            "-c:v",
            "copy",
            "-bsf:v",
            bsf_name,
            "-f",
            codec_name,
            "pipe:1",
        ]
        # Run ffmpeg in subprocess and redirect it's output to pipe
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        # Create HW decoder class
        nvdec = nvc.PyNvDecoder(w, h, f, c, g)
        cspace, crange = params['color_space'], params['color_range']
        cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

        # Initialize colorspace conversion chain
        if cspace != nvc.ColorSpace.BT_709:
            nvYuv = nvc.PySurfaceConverter(w, h, f, nvc.PixelFormat.YUV420, self.gpu_id)
        else:
            nvYuv = None

        if nvYuv:
            nvCvt = nvc.PySurfaceConverter(w, h, nvYuv.Format(), nvc.PixelFormat.RGB, self.gpu_id)
        else:
            nvCvt = nvc.PySurfaceConverter(w, h, f, nvc.PixelFormat.RGB, self.gpu_id)

        nvDwn = nvc.PySurfaceDownloader(w, h, nvCvt.Format(), self.gpu_id)

        # Amount of bytes we read from pipe first time.
        read_size = 4096
        # Total bytes read and total frames decded to get average data rate
        rt = 0
        fd = 0

        # Main decoding loop, will not flush intentionally because don't know the
        # amount of frames available via RTSP.
        t0 = time.time()
        print("running stream")
        while True:
            if length_seconds != 0:
                if (time.time() - t0) > length_seconds:
                    print(f"Listened for {length_seconds} seconds")
                    break
                # Pipe read underflow protection
            if not read_size:
                read_size = int(rt / fd)
                # Counter overflow protection
                rt = read_size
                fd = 1

                # Read data.
                # Amount doesn't really matter, will be updated later on during decode.
            bits = proc.stdout.read(read_size)
            if not len(bits):
                print("Can't read data from pipe")
                break
            else:
                rt += len(bits)

            # Decode
            enc_packet = np.frombuffer(buffer=bits, dtype=np.uint8)
            pkt_data = nvc.PacketData()
            try:
                surf = nvdec.DecodeSurfaceFromPacket(enc_packet, pkt_data)
                if not surf.Empty():
                    # To RGB + GPU-accelerated color conversion:
                    if nvYuv:
                        yuvSurface = nvYuv.Execute(surf, cc_ctx)
                        cvtSurface = nvCvt.Execute(yuvSurface, cc_ctx)
                    else:
                        cvtSurface = nvCvt.Execute(surf, cc_ctx)

                    if cvtSurface.Empty():
                        print('Failed to do color conversion')
                        continue
                    # Downloading image from GPU memory to numpy array:
                    rawFrame = np.ndarray(shape=(cvtSurface.HostSize()), dtype=np.uint8)
                    success = nvDwn.DownloadSingleSurface(cvtSurface, rawFrame)
                    if not success:
                        print('Failed to download surface')
                        continue
                    # RGB
                    img = rawFrame.reshape((h, w, 3))
                    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
                    key = 'frame_' + self.ip.split('@')[-1]
                    value = {'img': image2base64(img), 'now_time': now_time}
                    try:
                        # self.client.set(key, value)
                        self.redis_con.set(key,orjson.dumps(value))
                        print("set:", "key:", key, "now_time:", now_time)
                    except Exception as mem_e:
                        res = requests.post('http://' + self.web_ip + '/strategy_msg',
                                            json={'ip': ip.split('@')[-1], 'valid': False,
                                                  'model': '',
                                                  'msg': 'Memcached Set Error:' + str(mem_e) + '!',
                                                  'type': 'read_rtsp'})
                        print("Memcached Set Error:", mem_e)
                        return
                    fd += 1
                    # Shifts towards underflow to avoid increasing vRAM consumption.
                    if pkt_data.bsl < read_size:
                        read_size = pkt_data.bsl
                # else:
                #     time.sleep(0.04)
                    # nvdec = nvc.PyNvDecoder(w, h, f, c, g)


            # Handle HW exceptions in simplest possible way by decoder respawn
            except nvc.HwResetException as e: 
                print('Video Error:', e,247)
                nvdec = nvc.PyNvDecoder(w, h, f, c, g) 
                continue 
            except Exception as e:
                print('Video Error:', e)
                nvdec = nvc.PyNvDecoder(w, h, f, c, g)
                continue


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_ip', type=str, default='0', help='For connection.')

    # gpu decode
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU decode.')

    # for open camera fail info
    parser.add_argument('--web_ip', type=str, default='0.0.0.0:8000', help='Web ip.')

    # for memcached client ports
    parser.add_argument('--client_ports', type=str, default='11212;11213', help='Memcached port.')

    parser.add_argument('--listened_seconds', type=int, default=0, help='Memcached port.')
    opt = parser.parse_args()
    # print_args(vars(opt))
    return opt


def get_stream_params(url):
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout = proc.communicate()[0]

    bio = BytesIO(stdout)
    json_out = json.load(bio)

    params = {}
    if not "streams" in json_out:
        return {}

    for stream in json_out["streams"]:
        if stream["codec_type"] == "video":
            params["width"] = stream["width"]
            params["height"] = stream["height"]
            params["framerate"] = float(eval(stream["avg_frame_rate"])) if stream["avg_frame_rate"] != '0/0' else 25.0

            codec_name = stream["codec_name"]
            is_h264 = True if codec_name == "h264" else False
            is_hevc = True if codec_name == "hevc" else False
            if not is_h264 and not is_hevc:
                raise ValueError(
                    "Unsupported codec: "
                    + codec_name
                    + ". Only H.264 and HEVC are supported in this sample."
                )
            else:
                params["codec"] = (
                    nvc.CudaVideoCodec.H264 if is_h264 else nvc.CudaVideoCodec.HEVC
                )

                pix_fmt = stream["pix_fmt"]
                is_yuv420 = pix_fmt == "yuv420p"
                is_yuv444 = pix_fmt == "yuv444p"

                # YUVJ420P and YUVJ444P are deprecated but still wide spread, so handle
                # them as well. They also indicate JPEG color range.
                is_yuvj420 = pix_fmt == "yuvj420p"
                is_yuvj444 = pix_fmt == "yuvj444p"

                if is_yuvj420:
                    is_yuv420 = True
                    params["color_range"] = nvc.ColorRange.JPEG
                if is_yuvj444:
                    is_yuv444 = True
                    params["color_range"] = nvc.ColorRange.JPEG

                if not is_yuv420 and not is_yuv444:
                    raise ValueError(
                        "Unsupported pixel format: "
                        + pix_fmt
                        + ". Only YUV420 and YUV444 are supported in this sample."
                    )
                else:
                    params["format"] = (
                        nvc.PixelFormat.NV12 if is_yuv420 else nvc.PixelFormat.YUV444
                    )

                # Color range default option. We may have set when parsing
                # pixel format, so check first.
                if "color_range" not in params:
                    params["color_range"] = nvc.ColorRange.MPEG
                # Check actual value.
                if "color_range" in stream:
                    color_range = stream["color_range"]
                    if color_range == "pc" or color_range == "jpeg":
                        params["color_range"] = nvc.ColorRange.JPEG

                # Color space default option:
                params["color_space"] = nvc.ColorSpace.BT_601
                # Check actual value.
                if "color_space" in stream:
                    color_space = stream["color_space"]
                    if color_space == "bt709":
                        params["color_space"] = nvc.ColorSpace.BT_709

                return params
    return {}


if __name__ == '__main__':
    opt = vars(parse_opt())
    print(opt)
    print('开始摄像头视频解码')
    # time.sleep(5)

    ip = opt['camera_ip'].strip()

    # 判断摄像头是否可以打开
    cap = cv2.VideoCapture(ip)
    count = 0
    for i in range(12):
        if not cap.isOpened():
            count += 1
            time.sleep(5)
        else:
            break
    if count == 12:
        res = requests.post('http://' + opt['web_ip'] + '/strategy_msg',
                            json={'ip': ip.split('@')[-1], 'valid': False,
                                  'model': '',
                                  'msg': 'Failed to open!',
                                  'type': 'read_rtsp'})
        print(ip, "摄像头打开失败！")
        exit(0)

    # 是否使用GPU解码视频
    if opt['gpu_id'] == 'None':
        gpu_id = None
    else:
        gpu_id = int(opt['gpu_id'])

    rv = ReadVideo(ip, opt['client_ports'], gpu_id, opt['web_ip'])
    if gpu_id is not None:
        rv.read_by_gpu(length_seconds=opt['listened_seconds'])
    else:
        rv.read()