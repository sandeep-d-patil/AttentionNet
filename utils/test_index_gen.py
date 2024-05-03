import os
import argparse


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='test_index_generator')
    parser.add_argument('--seq_len', type=int, default=5, help='( Length of sequence of images')
    parser.add_argument('--stride', type=int, default=1, help='input batch size for training (default: 10)')
    parser.add_argument('--directory', type=str, default='/home/sandeep/PycharmProjects/Robust-Lane-Detection/testset/',
                        help='Directory of test dataset')
    args = parser.parse_args()
    return args


def test_file_generator(seq_len: int, stride: int, directory: str):
    """
    test_file_generator generates the locations of test files in string format.
    param seq_len: Length of sequence of images to be generated
    param stride: How many images have to skipped to generate next sequence
    param directory: Directory of the testset which has image, truth folders
    """
    image_folders = os.listdir(directory + 'image')
    truth_folders = os.listdir(directory + 'truth')
    seq_list = []
    for i, folder in enumerate(truth_folders):
        if folder in image_folders:
            truth_seq_list = sorted(os.listdir(directory + 'truth/' + folder), key=lambda x: int(x[:-4]))
            image_seq_list = sorted(os.listdir(directory + 'image/' + folder), key=lambda x: int(x[:-4]))
            j = 0
            while j < len(image_seq_list) and j + seq_len < len(image_seq_list):
                for k in range(seq_len):
                    seq_list.append(directory + 'image/' + folder + '/' + image_seq_list[j + k] + ' ')
                seq_list.append(directory + 'truth/' + folder + '/' + truth_seq_list[j + seq_len - 1])
                seq_list.append("\n")
                j += stride
    test_files = open("index_files/test_index.txt", "wt")
    for element in seq_list:
        test_files.write(element)
    test_files.close()


if __name__ == '__main__':
    # Initialize the arguments
    arg = args_setting()
    # Generate test_index.txt
    test_file_generator(arg.seq_len, arg.stride, arg.directory)

