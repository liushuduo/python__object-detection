import argparse
import cv2 as cv
from skimage import feature
import os
import pickle
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdir', help='Positive dataset path', default='./raw/training_data_48x48/pos')
    parser.add_argument('--ndir', help='Negative dataset path', default='./raw/training_data_48x48/neg')
    parser.add_argument('--output', help='Output dataset file', default='./data/HOG_dataset/train/training_data.pkl')
    return parser


def get_hog(img, opencv=False):
    """There are two methods of generating HOG descriptor -- using feature from skimage or HOGDescriptor from opencv. There is a slight difference. But I havn't figure out the reason. 

    Args:
        img (numpy.darray): [h, w, rgb]
        opencv (bool, optional): decide wether to use opencv HOG. Defaults to False.

    Returns:
        HOG Descriptor: HOG desciptor of given image.
    """
    if opencv:
        win_size = (48, 48)      # detect window size
        cell_size = (8, 8)       # cell size 
        block_size = (16, 16)    # block size, here it contains 4 blocks 
        block_stride = (8, 8)    # block stride is 8 pixels, i.e., 1 cell
        nbins = 9               # number of bins of orientations
        hog = cv.HOGDescriptor(win_size, block_size, block_stride,
                               cell_size, nbins)
        return hog.compute(img, winStride = (8,8), padding = (0,0))
        
    else:
        (ski_hog, ski_hog_image) = feature.hog(
            img, orientations=9, 
            pixels_per_cell=(8, 8), cells_per_block=(2,2), 
            block_norm='L2-Hys', visualize=True, transform_sqrt=True
        )
        return ski_hog


def get_data(train_data, label, path, labelType):
    
    for file_name in os.listdir(path):
        if file_name.endswith('.png') or file_name.endswith('jpg'):
            img_dir = os.path.join(path, file_name)
            
            # Get HOG Descriptor of image
            img = cv.imread(img_dir)
            hog_desc = get_hog(img)
            hog_desc = hog_desc.squeeze()
            
            # Add data to dataset
            label.append(labelType)
            train_data.append(hog_desc)
    
    return train_data, label


def get_dataset(pdir, ndir):
    train_data = []
    label = []
    
    # Generate positive sample
    train_data, label = get_data(train_data, label, pdir, 1)
    
    # Generate negative sample
    train_data, label = get_data(train_data, label, ndir, 0)

    return np.array(train_data, dtype=np.float32), \
            np.array(label, dtype=np.int32)


def main():
    parser = get_parser()
    args = parser.parse_args()
    train_data, label = get_dataset(args.pdir, args.ndir)
    
    with open(args.output, 'wb') as f:
        pickle.dump([train_data, label], f)


if __name__ == '__main__':
    main()