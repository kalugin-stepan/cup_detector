import cv2
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
import time
from utils import *

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

base_options = core.BaseOptions(
    file_name='cup.tflite', use_coral=False, num_threads=4)

detection_options = processor.DetectionOptions(
    max_results=1, score_threshold=0.5)
    
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)

detector = vision.ObjectDetector.create_from_options(options)

prev_frame_time = 0

while True:
    suc, img = cap.read()

    if not suc:
        break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imp_tensor = vision.TensorImage.create_from_array(rgb_img)

    detections = detector.detect(imp_tensor)

    rez = visualize(img, detections)

    cur_frame_time = time.time()

    fps = round(1/(cur_frame_time-prev_frame_time))

    rez = show_fps(rez, fps)

    prev_frame_time = cur_frame_time

    if cv2.waitKey(1) == 27:
        break

    cv2.imshow('rez', rez)

cap.release()
cv2.destroyAllWindows()