# 基於YOLOv11和MediaPipe的疲勞駕駛檢測

## 簡介
疲勞駕駛是近年交通事故的主要原因之一。
本專案結合 Mediapipe與YOLOv11，在即時影像中檢測駕駛是否有 長時間閉眼 或 頻繁打哈欠，並在檢測到疲勞行為時發出警示，降低意外風險。

___

## Result
**Demo Video**：

[![Demo Video](https://img.youtube.com/vi/1Dkxx64j-FE/0.jpg)](https://youtu.be/1Dkxx64j-FE)

<img src="./assets/Yawning_Detected.gif" width="800" height="600" />



## 專案結構

```
YOLODDD/
│── data/              # 資料集 (Kaggle, 眼睛/嘴巴裁切圖片)
│── models/            # YOLOv11 訓練權重
│── src/
│   ├── models/        # 主程式入口
│     ├── drowsy_detectors        # Mediapipe擷取+YOLO辨識及疲勞判定邏輯
│   ├── main_oop.py        # 主程式入口
│   ├── train.py    # YOLO模型訓練code
│── requirements.txt   # 相依套件
│── alert1             # 警告音檔
│── README.md          # 專案說明文件
```

## 安裝與執行

1. 複製專案
```
git clone https://github.com/DieWay0123/YOLODDD.git
cd YOLODDD
```
2. 安裝套件
```
pip install -r requirements.txt
```
3. 執行程式
```
# 需先將主程式內變數CAMERA_DEVICE_NUMBER調整為自己的camera編號
python src/main_oop.py
```

## 流程
使用MediaPipe偵測左右眼和嘴巴的位置 -> Crop出左右眼和嘴巴 -> 將左右眼和嘴巴分別餵給偵測兩者開合的模型偵測 -> 透過演算法進行疲勞檢測

## 備忘錄
1. 封裝擷取左右眼和嘴巴的函數
2. 測試對資料集用cv技術進行前處理前後的accuracy和loss

