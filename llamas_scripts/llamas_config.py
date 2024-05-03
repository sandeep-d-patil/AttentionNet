import argparse

# global param
# lane_dataset setting

img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 1
class_num = 2

# "./save/llamasresults/"
# "./save/new_results/"
# '/home/sandeep/PycharmProjects/LaneDetection/index_files/test_indexNL.txt'
# '/home/sandeep/PycharmProjects/LaneDetection/index_files/test_index_tvtlane.txt'
# '/home/sandeep/PycharmProjects/LaneDetection/index_files/test_llamas_index.txt'
# '/home/sandeep/PycharmProjects/LaneDetection/test_llamas_2001_12k.txt'
# "/home/sandeep/Robust-Lane-Detection/LaneDetectionCode/data/testfiles.txt"
# '/home/sandeep/PycharmProjects/Robust-Lane-Detection/test_index.txt'
# './index_files/tusimple_index.txt'

# path
train_path = 'index_files/testfiles.txt'
val_path = 'val_index.txt'
test_path = '/home/sandeep/PycharmProjects/LaneDetection/index_files/llamassample1389.txt'
save_path = 'TuSimple-test-save/'
pretrained_path = '/home/sandeep/PycharmProjects/LaneDetection/save/pretrainedpath/98.71590845387374llamas46.pth'
    # '/media/sandeep/B064373A64370320/Desktop/result/unetllamas-20210806T162834Z-001/unetllamas/97.03472055411318attllamas23.pth'
    # '/media/sandeep/B064373A64370320/Desktop/result/LaneDetectionparameter-20210806T163012Z-001/LaneDetectionparameter/save/results/97.4012476725886llamas14.pth'

# '/home/sandeep/Desktop/Lisa_jobs/LaneDetectionparameter/save/pretrainedpath/drive-download-20210728T111358Z-001/98.15645992854736encodedeattllamas18.pth'

# weight
class_weight = [0.02, 1.02]


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-Attention')
    parser.add_argument('--model', type=str, default='UNet-Attention', help='( UNet-Attention | UNet | ')
    parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
