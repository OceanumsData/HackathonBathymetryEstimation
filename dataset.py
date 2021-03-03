import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from scipy import ndimage


def random_transformation(im):
    """Randomly rotate or flip the image"""
    i = np.random.randint(8)
    if i == 0 :
        return im
    if i == 1 :
        return np.rot90(im, axes=(0,1), k=1)
    if i == 2 :
        return np.rot90(im, axes=(0,1), k=2)
    if i == 3 :
        return np.rot90(im, axes=(0,1), k=3)
    if i == 4:
        return np.flip(im, axis=1)
    if i == 5:
        return np.flip(np.rot90(im, axes=(0,1), k=1))
    if i == 6:
        return np.flip(np.rot90(im, axes=(0,1), k=2))
    if i == 7:
        return np.flip(np.rot90(im, axes=(0,1), k=3))

def quarter_subtile(im):
    i = np.random.randint(0,8)
    a = np.random.randint(0,20)
    b = np.random.randint(0,20)
    new_arr = np.zeros((40,40,4))
    if i==0:
        new_arr[a:a+20,b:b+20,:] = im[:20,:20,:]
        return new_arr
    if i==1:
        new_arr[a:a+20,b:b+20,:] = im[20:40,:20,:]
        return new_arr
    if i==2:
        new_arr[a:a+20,b:b+20,:] = im[20:40,20:40,:]
        return new_arr
    if i==3:
        new_arr[a:a+20,b:b+20,:] = im[:20,20:40,:]
        return new_arr
    else:
        return im

def reduce_image_dimension(image_to_reduce):
    image_reduced_dimension = (image_to_reduce.shape[0], image_to_reduce.shape[1])
    image_reduced = np.zeros(image_reduced_dimension)    
    for value in range (0,image_to_reduce.shape[2]):
        image_reduced = image_reduced + image_to_reduce[:,:,value]  
    return image_reduced

def auto_rotate(image):
    work_image = reduce_image_dimension(image)
    max_ratio_var = 0
    best_angle = 0
    for i in range(36):
        cur_image = ndimage.rotate(work_image, i*10, reshape=False)
        ratio_var = np.var(cur_image[19,:]) / (0.0001 + np.var(cur_image[:,19]))
        if ratio_var > max_ratio_var:
            max_ratio_var = ratio_var
            best_angle = i*10
    return ndimage.rotate(image, best_angle, reshape=False)

class HackathonDataset(Dataset):
    """Hackathon Dataset"""
    
    def __init__(self, csv_file, root_dir, use_raw=False, transform=None, auto_rotate=False):
        """
        Args:
            csv_file (string): Path to the csv file with paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.use_raw = use_raw
        self.auto_rotate = auto_rotate
        
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        img_file = self.file.iloc[idx]["image_file_name"]        
        if "22NCL" in img_file:
            img_dir= "guyane/guyane/"
        elif "28PCC" in img_file:
            img_dir = "saint_louis/saint_louis/"  
        elif "29SMD" in img_file:
            img_dir = "dataset_29SMD/dataset_29SMD/"  
        elif "29TNE" in img_file:
            img_dir = "dataset_29TNE/dataset_29TNE/"
        else:
            raise Exception('There is something wrong with image name')  
        image = np.load(self.root_dir + img_dir + img_file + ".npy")
        image = (image + 1) / 2  # Normalization
        if self.use_raw:
            image_raw = np.load(self.root_dir + img_dir + img_file + "_RAW.npy")
            image = np.concatenate((image, image_raw), axis=2)
        if self.auto_rotate:
            image = auto_rotate(image)
        elif self.transform:
            image = random_transformation(image)  # Add a random permutation of the image
        #image = quarter_subtile(image)
        image = np.moveaxis(image, -1, 0)  # Permute dimensions in order to have Cin, H, W instead of H, W, Cin
        image = image.astype(np.float32)  # We work with float (float32), not double (float64)
        target = self.file.iloc[idx]["z"]
        target = target.astype(np.float32)  # We work with float (float32), not double (float64)
        sample = {'image': image, 'z': target, "image_file_name": img_file}
        return sample