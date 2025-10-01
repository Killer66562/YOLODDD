from models import DrowsyDetector

import cv2
import os
import mediapipe as mp

from ultralytics import YOLO


LEFT_EYE_LANDMARKS = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LANDMARKS = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
MOUTH_LANDMARKS = [61, 291, 13, 14] #!ADD 13 14

INNER_LIP_LANDMARKS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

CAMERA_DEVICE_NUMBER = 0

DETECTION_MODELS_ROOT_PATH = "d_models"
PREDICTION_MODELS_ROOT_PATH = "models"

MODEL_TYPE_EYE = "eyes"
MODEL_TYPE_MOUTH = "mouth"

DETECTION_MODEL_FILENAME = "face_landmarker.task"
EYES_PREDICTION_MODEL_FILENAME = "model-20241225-014613.pt"
MOUTH_PREDICTION_MODEL_FILENAME = "model-20241224-142512.pt"

PICTURE_ROOT_PATH = "pictures"
DETECTION_LEFT_EYE_PICTURE_FILENAME = "left_eye.png"
DETECTION_RIGHT_EYE_PICTURE_FILENAME = "right_eye.png"
DETECTION_MOUTH_PICTURE_FILENAME = "mouth.png"

IMAGE_SAVE_PATH = "saved_images"


#!FIX: Detection model都用同一個 沒有分部位
DETECTION_MODEL_PATH = os.path.join(DETECTION_MODELS_ROOT_PATH, DETECTION_MODEL_FILENAME)
EYES_PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE_EYE, EYES_PREDICTION_MODEL_FILENAME)
MOUTH_PREDICTION_MODEL_PATH = os.path.join(PREDICTION_MODELS_ROOT_PATH, MODEL_TYPE_MOUTH, MOUTH_PREDICTION_MODEL_FILENAME)

#!ADD
DETECTION_LEFT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_LEFT_EYE_PICTURE_FILENAME)
DETECTION_RIGHT_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_RIGHT_EYE_PICTURE_FILENAME)
DETECTION_MOUTH_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, DETECTION_MOUTH_PICTURE_FILENAME)

#!ADD
PREDICTION_EYE_PICTURE_FILENAME = "YOLO_eye_pred_"
PREDICTION_EYE_PICTURE_PATH = os.path.join(PICTURE_ROOT_PATH, PREDICTION_EYE_PICTURE_FILENAME)

#Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICKNESS = 2
LINE_TYPE = 2

#Thres
YAWN_FRAME_THRES = 60
EYES_CLOSED_FRAME_THRES = 30
YAWN_COUNT_THRES = 10

# mediapipe臉部landmmark偵測 setting
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    num_faces=1, 
    base_options=BaseOptions(model_asset_path=DETECTION_MODEL_PATH), 
    running_mode=VisionRunningMode.IMAGE
)
# create instance of FaceLandmarker  
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)

eyes_model = YOLO(EYES_PREDICTION_MODEL_PATH)
mouth_model = YOLO(MOUTH_PREDICTION_MODEL_PATH)

#! Use if you have GPU support cuda
#eyes_model = eyes_model.cuda()
#mouth_model = mouth_model.cuda()

detector = DrowsyDetector(
    camera=cap, 
    face_detection_model=landmarker, 
    left_eye_landmarks=LEFT_EYE_LANDMARKS, 
    left_eye_kp_l=263, 
    left_eye_kp_r=362, 
    left_eye_kp_mid=473, 
    right_eye_landmarks=RIGHT_EYE_LANDMARKS, 
    right_eye_kp_l=33, 
    right_eye_kp_r=133, 
    right_eye_kp_mid=468, 
    eyes_offset=1.2, 
    mouth_landmarks=MOUTH_LANDMARKS, 
    mouth_kp_l=61, 
    mouth_kp_r=291, 
    mouth_offset=1.4, 
    eyes_classification_model=eyes_model, 
    eyes_closed_label="close eyes", 
    eyes_open_label="open eyes", 
    mouth_classification_model=mouth_model, 
    mouth_closed_label="no yawn", 
    mouth_open_label="yawn",
    lip_kp_up=13,
    lip_kp_down=14,
    face_landmarks=[10, 152],
    face_kp_up=10,
    face_kp_down=152,
    yawn_ratio_threshold=0.15
)

detector.run()

cap.release()
cv2.destroyAllWindows()