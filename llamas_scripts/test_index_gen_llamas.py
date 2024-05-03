import os
from torchvision import models
import argparse
import numpy as np


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='test_index_generator')
    parser.add_argument('--seq_len', type=int, default=5, help='( Length of sequence of images')
    parser.add_argument('--stride', type=int, default=15, help='input batch size for training (default: 1)')
    parser.add_argument('--directory', type=str, default='/media/sandeep/B064373A64370320/LLAMAS_dataset/',
                        help='Directory of test dataset')
    parser.add_argument('--sample_interval', type=list, default=[1, 3, 6, 10, 15])
    parser.add_argument('--split', type=str, default='valid', help='train or valid')
    args = parser.parse_args()
    return args


def test_file_generator(seq_len: int, stride: int, directory: str, array_list: list, split: str):
    """
    test_file_generator generates the locations of test files in string format.
    param seq_len: Length of sequence of images to be generated
    param stride: How many images have to skipped to generate next sequence
    param directory: Directory of the testset which has image, truth folders
    """
    image_folders = os.listdir(directory + 'resized/' + split + '/')
    truth_folders = os.listdir(directory + 'resized_target/' + split + '/')
    seq_list = []
    for i, folder in enumerate(truth_folders):
        if folder in image_folders:
            truth_seq_list = os.listdir(directory + 'resized_target/' + split + '/' + folder)
            image_seq_list = os.listdir(directory + 'resized/' + split + '/' + folder)
            j = 0
            sample = np.array(array_list)
            while j < len(image_seq_list) and j + sample[-1] < len(image_seq_list):
                for k in sample:
                    seq_list.append(directory + 'resized/' + split + '/' + folder + '/' + image_seq_list[j + k] + ' ')
                seq_list.append(directory + 'resized_target/' + split + '/' + folder + '/' + image_seq_list[j+sample[-1]][:-15] + '.png')
                seq_list.append("\n")
                j += stride
    test_files = open("index_files/llamassample1389.txt", "wt")
    for element in seq_list:
        test_files.write(element)
    test_files.close()


if __name__ == '__main__':
    # Initialize the arguments
    arg = args_setting()
    # Generate test_index.txt
    test_file_generator(arg.seq_len, arg.stride, arg.directory, arg.sample_interval, arg.split)
