import numpy as np
import cv2
import base64


def recap_frames(file_dir):
    """Recap median frame from sampled frames."""
    cap = cv2.VideoCapture(file_dir)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_get = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size = int(num_frames * 0.2))
    frames = []
    for i in frame_get:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, tmp = cap.read()
        if tmp is None:
            continue
        frames.append(tmp)
    cap.release()
    frame_median = np.median(frames, axis=0).astype(dtype=np.uint8)
    gray_frame_median = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)
    return gray_frame_median, num_frames

    
def process_video(file_dir):
    """Process whole video file and draw all detected recangle boxes."""
    gray_frame_median, num_frames = recap_frames(file_dir)
    cap = cv2.VideoCapture(file_dir)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    rect, frame = [], None
    for i in range(num_frames):
        _, tmp = cap.read()
        if tmp is None:
            continue
        frame = tmp
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_frame = cv2.absdiff(gray_frame, gray_frame_median)
        blur_frame = cv2.GaussianBlur(diff_frame, (5, 5), 0)
        _, thres_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thres_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1000:
            for j in contours:
                x, y, w, h = cv2.boundingRect(j)
                if (w * h > 64) & (y > int(height * 0.1)):
                    rect.append((x, y, w, h))
    for x, y, w, h in rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (123, 0, 255), 1)
    # cv2.imwrite(save_dir, frame) 
    retval, buffer = cv2.imencode('.jpg', frame)
    return str(int(len(rect) > 0)), base64.b64encode(buffer)
    

def process_video_gate(file_dir):
    """Process whole video file and draw all detected recangle boxes, specially for gate camera."""
    gray_frame_median, num_frames = recap_frames(file_dir)
    cap = cv2.VideoCapture(file_dir)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    rect, frame = [], None
    for i in range(num_frames):
        _, tmp = cap.read()
        if tmp is None:
            continue
        frame = tmp
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_frame = cv2.absdiff(gray_frame, gray_frame_median)
        blur_frame = cv2.GaussianBlur(diff_frame, (5, 5), 0)
        _, thres_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thres_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1000:
            for j in contours:
                x, y, w, h = cv2.boundingRect(j)
                x1, x2, y1, y2 = width * 0.28, width * 0.62, height * 0.66, height * 0.9
                if (w * h > 64) & ((x > x1) & (x < x2) & (y > y1) & (y < y2)):
                    rect.append((x, y, w, h))
    for x, y, w, h in rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (123, 0, 255), 1)
    # cv2.imwrite(save_dir, frame)   
    retval, buffer = cv2.imencode('.jpg', frame)
    return str(int(len(rect) > 0)), base64.b64encode(buffer)    