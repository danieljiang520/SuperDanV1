from ultralytics import YOLO
from PIL import Image
import cv2

def main():
    model = YOLO('../../ultralytics/runs/detect/train3/weights/best.pt') # select a .pt file in runs/detect/train #num/weights/best.pt
    results = model.track(source="https://www.youtube.com/watch?v=W4EGrNeFlys", show=True, save=True, conf=0.1) # save in runs/detect/track #num

if __name__ == "__main__":
    main()