from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('../../ultralytics/runs/detect/train6/weights/best.pt')  # select a .pt file in runs/detect/train #num/weights/best.pt

    # Validate the model
    metrics = model.val()

if __name__ == "__main__":
    main()