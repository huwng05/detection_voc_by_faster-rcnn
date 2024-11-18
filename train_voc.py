import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, RandomAffine

from datasets_voc import VOCDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, \
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection import mean_ap, MeanAveragePrecision
import argparse
import torch
import tqdm
import os
from pprint import pprint
import shutil

def get_args():
    parser = argparse.ArgumentParser("Train_VOC")
    parser.add_argument('--batch-size','-b',type=int,default=4)
    parser.add_argument('--num-workers','-n',type=int, default=15)
    parser.add_argument('--epoch','-e',type=int,default=50)
    parser.add_argument('--learning-rate','-l',type=float, default=1e-3)
    parser.add_argument('--momentum','-m',type=float,default=0.9)
    parser.add_argument("--tensorboard-path",'-t',type=str, default="../dataset/tensorboard")
    parser.add_argument('--save-path','-s', type=str, default="../dataset/model")
    parser.add_argument('--train-continue','-c',type=bool,default=False)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images,labels = zip(*batch)
    return list(images),list(labels)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = Compose([
        ToTensor(),
        RandomAffine(
            degrees=(-5,5),
            translate=(0.15, 0.15),
            scale=(0.15,0.85)
        ),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = VOCDataset(root='../dataset', year='2012', transforms=transform)
    dataset_val = VOCDataset(root='../dataset', year='2012',image_set='val', transforms=transform)
    pram = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'collate_fn': collate_fn
    }

    dataloader_train = DataLoader(dataset_train,**pram)
    dataloader_val = DataLoader(dataset_val,**pram)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=5).to(device)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(dataset_train.feature)+1).to(device)
    optimize = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    start_epoch = 0
    best_map = -1
    if args.train_continue:
        checkpoint = torch.load(os.path.join(args.save_path,'last.pt'))
        model.load_state_dict(checkpoint['model'])
        optimize.load_state_dict(checkpoint['optimize'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['map']
    model.to(device)

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.makedirs(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    for epoch in range(start_epoch,args.epoch):
        model.train()
        process_bar = tqdm.tqdm(dataloader_train)
        losses = []
        for iter,(images, labels) in enumerate(process_bar):
            # forward
            images = [image.to(device) for image in images]
            labels = [{'boxes':label['boxes'].to(device),'labels': label['labels'].to(device)} for label in labels]
            loss_dic = model(images,labels)
            loss = sum([los for los in loss_dic.values()])
            losses.append(loss.item())
            process_bar.set_description('Epoch: {}/{}, Loss: {:0.4f}'.format(epoch+1,args.epoch,np.mean(losses)))
            writer.add_scalar("Train/Loss", np.mean(losses), epoch*args.epoch + iter)

            # backward
            optimize.zero_grad()
            loss.backward()
            optimize.step()

        model.eval()
        process_bar = tqdm.tqdm(dataloader_val)
        metrics = MeanAveragePrecision(iou_type='bbox')
        with torch.no_grad():
            prediction = []
            for iter, (images, labels) in enumerate(process_bar):
                images = [image.to(device) for image in images]
                outputs = model(images,labels)
                outputs = [{"boxes": output['boxes'].to("cpu"), 'labels': output['labels'].to("cpu"),'scores': output['scores'].to("cpu")} for output in outputs]
                # for target in outputs:
                #     prediction.append({
                #         "boxes": target['boxes'].to("cpu"),
                #         "labels": target['labels'].to("cpu"),
                #         "scores": target["scores"].to("cpu")
                #     })
                metrics.update(outputs,labels)
            result = metrics.compute()
            writer.add_scalar("Val/mAP", result["map"], epoch)
            writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
            writer.add_scalar("Val/mAP_75", result["map_75"], epoch)

        best_map = result['map'] if best_map < result['map'] else best_map

        check_point = {
            'epoch': epoch+1,
            'model': model.state_dict(),
            'optimize': optimize.state_dict(),
            'map': result['map'],
            'best_map': best_map
        }

        torch.save(check_point, os.path.join(args.save_path,'last.pt'))
        if best_map <= result['map']:
            torch.save(check_point, os.path.join(args.save_path, 'best.pt'))


if __name__ == '__main__':
    args = get_args()
    train(args)

