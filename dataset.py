import numpy as np
from PIL import Image
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torch.utils.data.dataset import Dataset
from pre_processing import *


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image

class DataTrain(Dataset):
    def __init__(self, image_path, mask_path):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
        self.mask_arr = glob.glob(str(mask_path) + "\*")
        self.image_arr = glob.glob(str(image_path) + str("\*"))
        # Calculate len
        self.data_len = len(self.mask_arr)
        # calculate mean and stdev

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        """
        # GET IMAGE
        """
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name) #读某张图片
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img) # Convert the image into numpy array

        # Augmentation 数据增强
        # Brightness 亮度
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)
        #img_as_img.show()

        
        '''
        # Sanity Check for image
        img1 = Image.fromarray(img_as_np)
        img1.show()
        '''

        # Normalize the image
        img_as_np = normalization2(img_as_np, max=1, min=0) #转化为(0,1)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        #print(img_as_np.shape)
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        #msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255 # 结果只有两个值：0和1
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        #print(msk_as_tensor.shape)

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


class DataVal(Dataset):
    def __init__(self, image_path, mask_path):

        self.mask_arr = glob.glob(str(mask_path) + str("/*"))
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.mask_arr)

    def __getitem__(self, index):

        single_image = self.image_arr[index]
        img_as_img = Image.open(single_image)
        img_as_np = np.asarray(img_as_img)

        # Normalize the image
        img_as_np = normalization2(img_as_np, max=1, min=0) #转化为(0,1)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        #print(img_as_np.shape)
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        """
        # GET MASK
        """
        single_mask_name = self.mask_arr[index]
        msk_as_img = Image.open(single_mask_name)
        msk_as_np = np.asarray(msk_as_img)
        msk_as_np = msk_as_np/255

        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        return (img_as_tensor, msk_as_tensor)

    def __len__(self):

        return self.data_len

if __name__ == "__main__":

    train = DataTrain(
        ".\\data\\train\\images", ".\\data\\train\\masks") 

    #img, msk = train.__getitem__(0)
    #img, msk, origin= val.__getitem__(0)