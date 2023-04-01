from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('./runs/detect/train12/weights/best.pt')  # select a .pt file in runs/detect/train #num/weights/best.pt

    # Validate the model
    metrics = model.val()

if __name__ == "__main__":
    main()