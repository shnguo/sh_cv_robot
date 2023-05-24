
class StrategyConfig(BaseModel):
    iplist: List[str]
    scene: str


async def strategy_add(strtegy_config: StrategyConfig):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    success_ip_list = []
    for ip in strtegy_config.iplist:
        # print(body)
        
        try:
            subprocess.check_call(
                # f"ps aux | grep '[c]amera_ip {ip}' |  awk '{{print $2}}'  |  xargs sudo kill -9",
                f"kill -9 $(ps aux | grep 'send_post.py' | grep -E '[c]amera_ip rtsp:.*{ip} ' | grep -E 'model_type_map {scene_map} ' | awk '{{print $2}}')",
                shell=True)
        except Exception as e:
                    logger.error(e)       
        try:
            num_detection = subprocess.getoutput(f"ps aux | grep '[d]etection.py' | grep '{scene_map}'| wc -l")
            if int(num_detection)==0:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if (meminfo.free / 1024**2)<1500:
                    logger.error('not enough gpu memory 791')
                    return { 
                        'success_ip_list':success_ip_list,
                        'failed_ip_list':list(set(strtegy_config.iplist)-set(success_ip_list))
                    }
                shell_1 = f'''
                    /home/dl/miniconda3/bin/python detection.py --model_type_map '{scene_map}' >> ./logs/detection_{scene_map}.log  2>&1 &
                '''
                print(f'shell_1={shell_1}')    
                subprocess.check_call(
                    shell_1,
                    shell=True)
                time.sleep(1)
            async with app.state.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('''
                    select username,password from fst_ip_config where ip=%s
                    ''',(ip,))
                    u_p=await cur.fetchone()
            
            num_read_rtsp = subprocess.getoutput(f"ps aux | grep '[r]ead_rtsp.py' | grep '{ip} '| wc -l")
            if int(num_read_rtsp)==0:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if (meminfo.free / 1024**2)<400:
                    logger.error('not enough gpu memory 814')
                    return { 
                        'success_ip_list':success_ip_list,
                        'failed_ip_list':list(set(strtegy_config.iplist)-set(success_ip_list))
                    }
                try:
                    shell_2 = f'''
                        /home/dl/miniconda3/bin/python  read_rtsp.py --camera_ip 'rtsp://{u_p[0]}:{u_p[1]}@{ip}' \
                            --gpu_id 0 --web_ip '{app.state.ip}:8000' >> ./logs/read_rtsp_{ip}.log 2>&1 &
                    '''
                    print(f'shell_2={shell_2}')    
                    subprocess.check_call(
                        shell_2,
                        shell=True)
                    # time.sleep(1)
                except Exception as e:
                    logger.error(e)
            shell_str = f'''/home/dl/miniconda3/bin/python send_post.py  --camera_ip 'rtsp://{u_p[0]}:{u_p[1]}@{ip}' \
            --scene '{scene}' \
            >> ./logs/send_post_{ip}.log 2>&1 &''',
            print(f'shell_str={shell_str}')    
            subprocess.check_call(
                shell_str,
                shell=True)
        except Exception as e:
                    logger.error(e)