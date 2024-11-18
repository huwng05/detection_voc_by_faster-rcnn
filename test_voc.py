import torch
import argparse
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn



def get_args():
    parser = argparse.ArgumentParser(description="Test faster r-cnn")
    parser.add_argument('--image-path','-p',type=str, help='path to image',required=True)
    parser.add_argument('--best-model','-b',type=str, default='../dataset/model/best.pt')
    parser.add_argument("--conf-threshold",'-c', type=float,default=0.05)
    args = parser.parse_args()
    return args

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)

    if args.best_model:
        checkpoint = torch.load(args.best_model)
        model.load_state_dict(checkpoint['model'])

    model.float()
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image,(2,0,1))/255.
    image = [torch.from_numpy(image).float()]

    categories = ["person","bird", "cat", "cow", "dog", "horse", "sheep",
                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor" ]
    with torch.no_grad():
        model.eval()
        output = model(image)[0]
        bboxes = output['boxes']
        labels = output['labels']
        scores = output['scores']
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > args.conf_threshold:
                xmin,ymin,xmax,ymax = bbox
                cv2.rectangle(ori_image,(int(xmin),int(ymin)), (int(xmax),int(ymax)),(0,0,255))
                cv2.putText(ori_image, categories[label], (int(xmin), int(ymin)), cv2.FONT_ITALIC, 1.5,(0,255,0), cv2.LINE_4)
        cv2.imwrite("prediction.jpg", ori_image)

if __name__ == '__main__':
    args = get_args()
    test(args)