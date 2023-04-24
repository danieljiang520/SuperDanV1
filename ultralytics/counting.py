from ultralytics import YOLO
from PIL import Image
import torch
import sys


def preprocess_result(results):
    boxes = []

    for r in results:
        box = r.boxes
        conf = box.conf
        xyxy = box.xyxy
        if len(conf) == 0:
            continue
        
        # pick the xyxy with highest confidence
        conf_max_xyxy = xyxy[torch.argmax(conf)]

        # center of the bounding box (x, y)
        center = torch.tensor([(conf_max_xyxy[0] + conf_max_xyxy[2]) / 2, (conf_max_xyxy[1] + conf_max_xyxy[3])  / 2])
        boxes.append(center)
    boxes = torch.stack(boxes, dim=0)
    return boxes


def counting_return(boxes):
    # 1 is going up, 0 is going down

    net_height = torch.mean(boxes[:, 1])
    print(net_height)

    count = 0
    # -1 is down, 1 is up area
    curr_side = -1 if boxes[0][1] < net_height else 1
    for box in boxes:
        if box[1] > net_height:
            side = 1
        else:
            side = 0

        if side != curr_side:
            count += 1
            curr_side = side
    return count

def print_help():
    print("Usage: python counting.py [.pt file location] [source video location] [confidence]")
    
    
def main():
    if len(sys.argv) != 4:
        print_help()
        return
    _, model_location, source, conf = sys.argv
    if model_location == None or source == None or conf == None:
        print("Usage: python counting.py [.pt file location] [source video location] [confidence]")
    model = YOLO(model_location) # select a .pt file in runs/detect/train #num/weights/best.pt
    results = model.predict(source=source, conf=conf)
    boxes = preprocess_result(results)
    count = counting_return(boxes)
    print(count)


if __name__ == "__main__":
    main()

