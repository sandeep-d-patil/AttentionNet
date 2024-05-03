import torch
import configs.config as config
from configs.config import args_setting
from models.model import generate_model
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image


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
        sample = {"data": data, "label": label, "target_path": img_path_list[5]}
        return sample


def output_result(model, test_loader, device):
    """

    :param model :
    :param test_loader :
    :param device :
    """
    model.eval()
    k = 0
    feature_dic = []
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            k += 1
            print(k)
            data, target = sample_batched["data"].to(device), sample_batched[
                "label"
            ].type(torch.LongTensor).to(device)
            # data, target = x.to(device), y.type(torch.LongTensor).to(device)
            output, feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == "SegNet-ConvLSTM" or "UNet-ConvLSTM":
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255

            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = img.getpixel((i, j))
                    if img2[0] > 200 or img2[1] > 200 or img2[2] > 200:
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(
                config.save_path + "%s_data.jpg" % k
            )  # red line on the original image
            img.save(config.save_path + "%s_pred.jpg" % k)  # prediction result


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


if __name__ == "__main__":
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    test_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList(file_path=config.test_path, transforms=op_tranforms),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=1,
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

    # store the output result pictures
    output_result(model, test_loader, device)
