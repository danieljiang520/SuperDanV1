from ultralytics import YOLO
from PIL import Image
import cv2

def main():
    model = YOLO('./runs/detect/train12/weights/best.pt') # select a .pt file in runs/detect/train #num/weights/best.pt
    results = model.track(source="https://www.youtube.com/watch?v=W4EGrNeFlys", show=False, save=True, conf=0.3) # save in runs/detect/track #num

if __name__ == "__main__":
    main()