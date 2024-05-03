import torch
from data.dataset_new import RoadSequenceDatasetList
from torchvision import transforms
import cv2
from models.model import UNet_Attention
import matplotlib

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

matplotlib.use("TkAgg")


def image_overlay_second_method(
    img1, img2, location, min_thresh=0, is_transparent=False
):
    h, w = img1.shape[:2]
    h1, w1 = img2.shape[:2]
    x, y = location
    roi = img1[y : y + h1, x : x + w1]

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img_bg, img_fg)
    if is_transparent:
        dst = cv2.addWeighted(img1[y : y + h1, x : x + w1], 0.1, dst, 0.9, None)
    img1[y : y + h1, x : x + w1] = dst
    return img1


def show_activations(input, hidden_output, filename, orig_image):
    input = input.cpu().detach().numpy().reshape(720, 1280, 3)
    # input = cv2.resize(input, (256, 128), cv2.INTER_CUBIC)
    input_gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    hidden_output = hidden_output.reshape(8, 16)
    hidden_output_resize = cv2.resize(hidden_output, (1280, 720), cv2.INTER_CUBIC)
    plot_1 = 0.005 * hidden_output_resize + input_gray
    os.makedirs(
        "/home/sandeep/attention_activations_index2/" + filename[:-2], exist_ok=True
    )
    plt.imshow(plot_1)
    plt.savefig(
        "/home/sandeep/attention_activations_index2/"
        + filename[:-2]
        + f"/attention_sum_{filename}_1.png"
    )


def unravel_list(list_of_tensor):
    list_of_array = []
    for i in list_of_tensor:
        list_of_array.append(i.detach().cpu().numpy())
    return list_of_array


def main_loop():

    device = torch.device("cuda")
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        RoadSequenceDatasetList("test_tasks_0627.json", transforms=op_tranforms),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    model = UNet_Attention(3, 2).to(device)
    class_weight = torch.Tensor([0.02, 1.02])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    pretrained_dict = torch.load(
        "/home/sandeep/PycharmProjects/LaneDetection/save/pretrainedpath/98.2323018503801.pth"
    )
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader, start=20):
            data, target, orig_img_data = (
                sample_batched["data"].to(device),
                sample_batched["label_name"][0],
                sample_batched["orig_image"],
            )
            print(target)
            orig_image = sample_batched["orig_image"][0][-1]
            y = model(data)
            attention_output_num = unravel_list(y[2])
            pred = y[0].max(1, keepdim=True)[1]
            img2 = (
                torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            )
            final_img = Image.fromarray(img2.astype(np.uint8))
            final_img = final_img.resize((1280, 720), Image.BOX)
            orig_image = torch.squeeze(orig_image).cpu().numpy()
            orig_image = np.transpose(orig_image, [1, 2, 0]) * 255
            orig_image = Image.fromarray(orig_image.astype(np.uint8))
            rows = final_img.size[0]
            cols = final_img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img3 = final_img.getpixel((i, j))
                    if img3[0] > 0 or img3[1] > 0 or img3[2] > 0:
                        orig_image.putpixel((i, j), (234, 53, 57, 255))

            for i, (input, att) in enumerate(
                zip(orig_img_data[0], attention_output_num)
            ):
                show_activations(input, att, target + f"_{i}", orig_image)
            orig_image.save(
                "/home/sandeep/attention_activations_index2/"
                + target
                + "/output_"
                + target
                + f"_{i}"
                + ".png"
            )


if __name__ == "__main__":
    main_loop()
