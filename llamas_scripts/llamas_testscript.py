import torch
from models.model import generate_model
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models
from utils import *
import torch.nn as nn
import os

from unsupervised_llamas.evaluation.evaluate_segmentation import (
    binary_eval_single_image,
    evaluate_set,
)


def readTxt(file_path):
    img_list = []
    with open(file_path, "r") as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list


class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):
        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(
                torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0)
            )
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {"data": data, "label": label, "image_path": img_path_list[4]}
        return sample


torch.manual_seed(42)

device = torch.device("cuda")

op_tranforms = transforms.Compose([transforms.ToTensor()])

test_loader = torch.utils.data.DataLoader(
    RoadSequenceDatasetList(
        file_path="/index_files/test_index_llamas1.txt", transforms=op_tranforms
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = downfirst(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.inconv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
        self.outconv = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1))
        self.outc = outconv(64, n_classes)
        self.attention_module = AttentionModule(input_size=128, hidden_size=128)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.inconv(x5)
            data.append(x6.unsqueeze(0))
        data = torch.cat(data, dim=0)
        data = torch.flatten(data, start_dim=2)
        test, _ = self.attention_module(data)
        output_tensor = test.permute(1, 0, 2)
        output_tensor_new = output_tensor.reshape(
            output_tensor.shape[0], output_tensor.shape[1], 8, 16
        )
        output = self.outconv(output_tensor_new)
        x = self.up1(output, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, test


class_weight = torch.Tensor([0.02, 1.02])

criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

model = UNet_ConvLSTM(3, 2).to(device)
# import os
# root_path = '/home/sandeep/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/'#'/home/sandeep/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/'
# files = os.listdir(root_path)
# for pretrained_path in files:

pretrained_dict = torch.load(
    "/home/sandeep/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/97.4012476725886llamas14.pth"
)
model_dict = model.state_dict()
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
model_dict.update(pretrained_dict_1)
model.load_state_dict(model_dict)


dataset = next(iter(test_loader))
data, label, path = dataset["data"], dataset["label"], dataset["image_path"]

k = 0
roota_path = "/save/LLAMAS_dataset"
with torch.no_grad():
    for batch_idx, sample_batched in enumerate(test_loader):
        k += 1
        data, target, path = (
            sample_batched["data"].to(device),
            sample_batched["label"].type(torch.LongTensor).to(device),
            sample_batched["image_path"],
        )
        # data, target = x.to(device), y.type(torch.LongTensor).to(device)
        output, feature = model(data)
        pred = output.max(1, keepdim=True)[1]
        img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
        # image_scaled = cv2.resize(img.astype(np.uint8), (1276, 717), cv2.INTER_CUBIC)
        # kernel = np.ones((3, 3), np.uint8)
        # img_erosion = cv2.erode(img, kernel, iterations=1)
        img_save = Image.fromarray(img.astype(np.uint8))
        if not os.path.exists(roota_path + "/" + path[0][33:-36]):
            os.makedirs(roota_path + "/" + path[0][33:-36])
        img_save.save(roota_path + "/" + path[0][33:-15] + ".json_1.png")

# inference_folder = '/home/sandeep/PycharmProjects/LaneDetection/save/LLAMAS_dataset/LAMAS_dataset/resized/valid/'
# dataset_split = 'images-2014-12-22-14-19-07_mapping_280S_3rd_lane'
# max_workers = 8
# # print(pretrained_path)
# eval_function = binary_eval_single_image
# evaluate_set(inference_folder, eval_function, dataset_split,
#              max_workers=max_workers)

# import os
# import shutil
# target_files_root = os.listdir(
#     '/home/sandeep/PycharmProjects/LaneDetection/save/LAMAS_eval/LAMAS_dataset/resized/valid/images-2014-12-22-14-19-07_mapping_280S_3rd_lane')
# root_path = '/home/sandeep/PycharmProjects/LaneDetection/save/LAMAS_eval/LAMAS_dataset/resized/valid/images-2014-12-22-14-19-07_mapping_280S_3rd_lane'
# source_folder = '/media/sandeep/Files/LAMAS_dataset/labels/valid/images-2014-12-22-14-19-07_mapping_280S_3rd_lane'
# for file in target_files_root:
#     shutil.copy(source_folder + '/' + file[:-6], root_path + '/' + file[:-6])
