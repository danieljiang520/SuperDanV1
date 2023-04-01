from ultralytics import YOLO

def main():
    # Load a model
    #model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='../datasets/data.yaml', epochs=10, imgsz=640)

if __name__ == "__main__":
    main()

