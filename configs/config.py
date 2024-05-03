import argparse


img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 1
class_num = 2

# path
train_path = "data/train_index.txt"
val_path = "data/val_index.txt"
test_path = "data/testfiles.txt"
save_path = "results/"
pretrained_path = "pretrained/98.2323018503801.pth"

# weight
class_weight = [0.02, 1.02]


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch UNet-Attention")
    parser.add_argument(
        "--model", type=str, default="UNet_Attention", help="( UNet_Attention | UNet "
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Use pretrained model",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If True, evaluate model else output result",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="use CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()
    return args
