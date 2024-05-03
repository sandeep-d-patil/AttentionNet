import numpy as np
import torch
import tqdm
import os
import sys
import inspect
import llamas_config
from llamas_config import args_setting
from torchvision import transforms
import numpy
from torch.utils.data import Dataset
from PIL import Image
import pprint
import cv2

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.model import generate_model


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
        label = np.array(Image.open(img_path_list[5]).convert("L"))
        # label = torch.squeeze(self.transforms(label))
        sample = {"data": data, "label": label, "target_path": img_path_list[5]}
        return sample


class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = torch.unsqueeze(self.transforms(data), dim=0)
        label = torch.squeeze(self.transforms(label))
        sample = {"data": data, "label": label}
        return sample


def thresholded_binary(prediction, threshold):
    """Thresholds prediction to 0 and 1 according to threshold"""
    threshold_value = torch.nn.Threshold(threshold, value=prediction.max())
    # print('max-pred', prediction.max())
    # print('min-pred', prediction.min())
    pred_threshold = threshold_value(prediction)
    pred = pred_threshold.max(1, keepdim=True)[1]

    # print('pred_shape', pred.shape)
    # print("unique pred", np.unique(pred.cpu().numpy(), return_counts=True))
    pred = torch.squeeze(pred).cpu().numpy().reshape(128, 256, 1)

    pred = cv2.dilate(pred.astype(np.uint8), kernel=(3, 3), iterations=1)
    # cv2.imshow('pred', 255 * pred.astype(np.uint8))
    # cv2.waitKey(20)
    return pred.reshape(128, 256, 1).astype(np.int32)


def true_positive(prediction, label):
    """Calculates number of correctly classified foreground pixels"""
    # print("unique pred", np.unique(prediction, return_counts=True))
    # print("unique label", np.unique(label, return_counts=True))
    num_tp = numpy.sum(
        numpy.logical_and(
            label.astype(np.int32) != 0,
            numpy.logical_and(prediction <= label.astype(np.int32), prediction > 0),
        )
    )
    # print(num_tp)
    return num_tp


def false_positive(prediction, label):
    """Calculates number of incorrectly predicted foreground pixels"""
    num_fp = numpy.sum(numpy.logical_and(label == 0, prediction != 0))
    return num_fp


def true_negative(prediction, label):
    """Calculates number of correctly identified background pixels"""
    num_tn = numpy.sum(numpy.logical_and(label == 0, prediction == label))
    return num_tn


def false_negative(prediction, label):
    """Calculates number of missed foreground pixels"""
    num_fn = numpy.sum(numpy.logical_and(label != 0, prediction == 0))
    return num_fn


def binary_approx_auc(prediction, label):
    """Calculates approximated auc and best precision-recall combination

    Parameters
    ----------
    prediction : numpy.ndarray
                 raw prediction output in [0, 1]
    label : numpy.ndarray
            target / label, values are either 0 or 1

    Returns
    -------
    Dict of approximate AUC, "corner" precision, "corner" recall
    {'precision', 'recall', 'auc'}

    Notes
    -----
    See docstring for alternative implementation options
    Approximated by 100 uniform thresholds between 0 and 1
    """
    # NOTE May achieve speedup by checking if label is all zeros
    num_steps = 100
    auc_value = 0

    # Most upper right precision, recall point
    corner_precision = 0
    corner_recall = 0
    corner_auc = 0
    corner_threshold = 0

    precisions = [1]
    recalls = [0]
    min_value = prediction.min()
    # print("min_value", min_value)
    max_value = prediction.max()
    # print("max_value", max_value)
    thresh_range = np.linspace(min_value.cpu().numpy(), 0, 100)

    # Individual precision recall evaluation for those steps
    for i in thresh_range:
        # threshold = ((num_steps - i) / 10) - 14

        thresholded_prediction = thresholded_binary(prediction, i)

        # tn = true_negative(thresholded_prediction, label)
        tp = true_positive(thresholded_prediction, label)
        fn = false_negative(thresholded_prediction, label)
        fp = false_positive(thresholded_prediction, label)

        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

        if (precision * recall) > corner_auc:
            corner_auc = precision * recall
            corner_precision = precision
            corner_recall = recall
            corner_threshold = i

        precisions.append(precision)
        recalls.append(recall)

        auc_value += (recalls[-1] - recalls[-2]) * precisions[-2]

    results = {
        1: {
            "recall": corner_recall,
            "precision": corner_precision,
            "threshold": corner_threshold,
            "auc": auc_value,
        }
    }

    return results


def get_parameters(model, layer_name):
    """ """
    import torch.nn as nn

    modules_skipped = (nn.ReLU, nn.MaxPool2d, nn.Dropout2d, nn.UpsamplingBilinear2d)
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma


def evaluate_model(model, test_loader, device, criterion):
    """ """
    model.eval()
    i = 0
    eval_dicts = {}
    length = np.arange(len(test_loader))
    # print(length)
    with torch.no_grad():
        for batch_idx, sample_batched in tqdm.tqdm(
            zip(length, test_loader),
            desc="Scoring test samples",
            total=len(test_loader),
        ):
            i += 1
            data, target = sample_batched["data"].to(device), sample_batched[
                "label"
            ].to(device)
            pred, _ = model(data)
            output = pred.max(1, keepdim=True)[1]
            img = (
                torch.squeeze(output).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            )
            img = img.astype(np.uint8)
            # cv2.imshow('img', img)
            # cv2.waitKey(2000)
            # pred = torch.squeeze(output)
            # print(pred.shape)
            # print(target.shape)
            target = target.cpu().numpy().reshape(128, 256, 1)
            # cv2.imshow('target', target)
            # cv2.waitKey(2000)

            for g in range(target.shape[0]):
                for j in range(target.shape[1]):
                    if target[g, j] > 0:
                        target[g, j] = 1
            # print("unique target", np.unique(target, return_counts=True))
            # print('min', min(target))
            # print('max', max(target))

            eval_dicts[batch_idx] = binary_approx_auc(prediction=pred, label=target)
        # The reduce step. Calculates averages
        label_paths = list(eval_dicts.keys())
        lanes = list(eval_dicts[label_paths[0]].keys())
        # print(eval_dicts)
        metrics = list(eval_dicts[label_paths[0]][lanes[0]].keys())

        mean_results = {}
        for lane in lanes:
            mean_results[lane] = {}
            for metric in metrics:
                mean = 0
                for label_path in label_paths:
                    mean += eval_dicts[label_path][lane][metric]
                mean /= len(label_paths)
                mean_results[lane][metric] = mean

        pprint.pprint(mean_results)
        return mean_results


if __name__ == "__main__":
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    test_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(
            file_path=llamas_config.test_path, transforms=op_tranforms
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
    )

    # load model and weights
    model = generate_model(args)
    class_weight = torch.Tensor(llamas_config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    # root_path = '/media/sandeep/B064373A64370320/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/'#'/home/sandeep/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/'
    # files = os.listdir(root_path)
    # for pretrained_path in files:
    #     print(pretrained_path)
    pretrained_dict = torch.load(llamas_config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # store the output result pictures
    evaluate_model(model, test_loader, device, criterion)
