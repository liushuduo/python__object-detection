from typing import Tuple
from skimage import feature
import cv2 as cv
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Get HOG image of the input image.')
    parser.add_argument('--file', help='Input file name.', default='./data/training-data/pos/img000.jpg')
    parser.add_argument('--dest', help='Path of output file.', default='./result')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_file = args.file 
    filename = str.split(input_file, '/')[-1]

    image = cv.imread(input_file)
    
    (hog, hog_image) = feature.hog(
        image, orientations=9, 
        pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
        block_norm='L2-Hys', visualize=True, transform_sqrt=True
    )
    print(hog)
    print(hog.shape)
    cv.imwrite(args.dest+'/hog_'+filename, hog_image*255)


if __name__ == '__main__':
    main()