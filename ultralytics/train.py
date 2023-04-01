from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(data='./datasets/data.yaml', epochs=10, imgsz=640)

if __name__ == "__main__":
    main()

