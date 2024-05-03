import torch
import configs.config as config
from configs.config import args_setting
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from dataset.dataset import RoadSequenceDataset, RoadSequenceDatasetList
from models.model import generate_model

PRINT_MODEL_PARAMS = False
SAVE_IMAGE = False


def output_result(model, test_loader, device):
    """

    :param model :
    :param test_loader :
    :param device :
    """
    model.eval()
    k = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            k += 1
            print(k)
            data, target = sample_batched["data"].to(device), sample_batched[
                "label"
            ].type(torch.LongTensor).to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            img = Image.fromarray(img.astype(np.uint8))
            data = torch.squeeze(data).cpu().numpy()
            if args.model == "UNet_Attention":
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            # data = cv2.erode(data, kernel=np.ones((5,5)), iterations=3)
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = img.getpixel((i, j))
                    if img2[0] > 0 or img2[1] > 0 or img2[2] > 0:
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data = data.resize((1280, 720), Image.LANCZOS)
            data.save(
                config.save_path + "%s_data.jpg" % k
            )  # red line on the original image


def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    k = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for sample_batched in test_loader:
            k += 1
            print(k)
            data, target = sample_batched["data"].to(device), sample_batched[
                "label"
            ].type(torch.LongTensor).to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().numpy() * 255
            lab = torch.squeeze(target).cpu().numpy() * 255
            img = img.astype(np.uint8)  # for pred_recall
            lab = lab.astype(np.uint8)  # for label_precision
            kernel = np.uint8(np.ones((3, 3)))

            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)

            a = len(np.nonzero(img * label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b == 0:
                error = error + 1
                continue
            else:
                fp += float(b - a)
                precision += float(a / b)
            c = len(np.nonzero(pred_recall * lab)[1])
            d = len(np.nonzero(lab)[1])

            if d == 0:
                error = error + 1
                continue
            else:
                fn += float(d - c)
                recall += float(c / d)
            F1_measure = (2 * precision * recall) / (precision + recall)
    test_loss /= len(test_loader.dataset) / args.test_batch_size
    test_acc = (
        100.0
        * int(correct)
        / (len(test_loader.dataset) * config.label_height * config.label_width)
    )
    print(
        "\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)".format(
            test_loss, int(correct), len(test_loader.dataset), test_acc
        )
    )

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)

    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print(
        "Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n".format(
            precision, recall, F1_measure
        )
    )
    evaluate_result = {
        "precision": precision,
        "recall": recall,
        "F1_measure": F1_measure,
        "test_acc": test_acc,
    }
    return evaluate_result


def get_parameters(model, layer_name):
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


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lane_img, target_lanes):
        for t in self.transforms:
            lane_img, target_lanes = t(lane_img), t(target_lanes)

        return lane_img, target_lanes


if __name__ == "__main__":
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if args.model == "UNet_Attention":
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(config.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=1,
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=config.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=1,
        )

    transform = Compose(
        [
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
        ]
    )

    # load model and weights
    model = generate_model(args)
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # To calculate the number of parameters of the model
    if PRINT_MODEL_PARAMS:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        from thop import profile

        flops, params = profile(model, inputs=(input.to(device)))
        print(flops)
        print(params)

    if args.evaluate:
        # calculate the values of accuracy, precision, recall, f1_measure
        evaluate_model(model, test_loader, device, criterion)
    else:
        # output the result pictures
        output_result(model, test_loader, device)
