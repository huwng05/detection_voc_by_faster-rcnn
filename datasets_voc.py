from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
import torch

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set="train", download = False, transforms=ToTensor()):
        super().__init__(root, year, image_set, download, transform=transforms)
        self.feature = features = ["person","bird", "cat", "cow", "dog", "horse", "sheep",
                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor" ]

    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        boxes = []
        labels = []
        for bbox in label['annotation']['object']:
            box = [int(bbox['bndbox']['xmin']),
                   int(bbox['bndbox']['ymin']),
                   int(bbox['bndbox']['xmax']),
                   int(bbox['bndbox']['ymax'])]
            target = self.feature.index(bbox['name'])
            boxes.append(box)
            labels.append(target)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        label = {
            'boxes': boxes,
            'labels':labels
        }
        return image, label
