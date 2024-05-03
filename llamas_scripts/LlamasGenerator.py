import os
import numpy as np
from PIL import Image
import PIL

from unsupervised_llamas.label_scripts.visualize_labels import create_segmentation_image


def image_list(dir_list, images: bool, folder: str):
    """

    :param dir_list:
    :param folder:
    :return:
    """
    image_lists = []
    folder_lists = []
    if images:
        folder_name = 'grayscale_images'
        resized_name = 'resized'
    else:
        folder_name = 'labels'
        resized_name = 'resized_target'
    for i in dir_list:
        if not os.path.exists(root_path + resized_name + '/' + folder + '/' + i):
            os.makedirs(root_path + resized_name + '/' + folder + '/' + i)
        files = os.listdir(root_path + folder_name + '/' + folder + '/' + i)
        for j in files:
            image_lists.append(root_path + folder_name + '/' + folder + '/' + i + '/' + j)
            folder_lists.append(folder + '/' + i + '/' + j)
    return image_lists, folder_lists


def gen_images(root_path: str, train: bool, images: bool):
    # Folders of grayscale_images
    train_files = os.listdir(root_path + 'grayscale_images')

    # Folders of label files
    label_files = os.listdir(root_path + 'labels')

    # List of folders in grayscale_images folder
    image_train_folders = [os.listdir(root_path + 'grayscale_images/' + train_file) for train_file in train_files]

    # Create list of train and validation images
    train_images, folder_train_lists = image_list(image_train_folders[1], images=True, folder='train')
    val_images, folder_valid_lists = image_list(image_train_folders[2], images=True, folder='valid')

    label_train_folders = [os.listdir(root_path + 'labels/' + label_file) for label_file in label_files]
    train_labels, folder_label_train_list = image_list(label_train_folders[0], images=False, folder='train')

    # label_train_folders = [os.listdir(root_path + 'labels/' + label_file) for label_file in label_files]
    valid_labels, folder_label_valid_list = image_list(label_train_folders[1], images=False, folder='valid')

    if train and images:
        for image, save_loc in zip(train_images, folder_train_lists):
            image_ex = Image.open(image)
            image_ex = image_ex.resize((256, 128), PIL.Image.LANCZOS).convert('RGB')
            name = 'resized/' + image[48:]
            image_ex.save(root_path + name, 'PNG')

    if not train and images:
        for label, save_loc in zip(train_labels, folder_label_train_list):
            image_ex = create_segmentation_image(label,color=(255,255,255), image='blank')
            image_ex = Image.fromarray(image_ex.astype(np.uint8))
            image_ex = image_ex.resize((256, 128), PIL.Image.LANCZOS).convert('RGB')
            name = 'resized_target/' + label[54:-5] + '.png'
            image_ex.save(os.path.join(root_path, name), 'PNG')

    if not train and not images:
        for label, save_loc in zip(valid_labels, folder_label_valid_list):
            image_ex = create_segmentation_image(label, color=(255, 255, 255), image='blank')
            image_ex = Image.fromarray(image_ex.astype(np.uint8))
            image_ex = image_ex.resize((256, 128), PIL.Image.LANCZOS).convert('L')
            for i in range(image_ex.size[0]):
                for j in range(image_ex.size[1]):
                    if image_ex.getpixel((i, j))>0:
                        image_ex.putpixel((i, j), 255)
            name = 'resized_target/' + label[54:-5] + '.png'
            image_ex.save(os.path.join(root_path, name), 'PNG')


if __name__ == '__main__':
    # Root folder unsupervised_llamas dataset where labels and images are stored
    root_path = '/media/sandeep/B064373A64370320/LLAMAS_dataset/'

    gen_images(root_path,train=True, images=True)
