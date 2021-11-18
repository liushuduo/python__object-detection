import argparse
import cv2 as cv
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/training-data/neg/img000.jpg')
    parser.add_argument('--output', default='./data/training-data-48x48/neg')
    parser.add_argument('--stride', nargs='+', type=int, default=[16, 16])
    parser.add_argument('--window_size', nargs='+', type=int, default=[48, 48])
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    stride = args.stride
    window_size = args.window_size

    bg = cv.imread(args.input)
    height, width = bg.shape[:2]

    # dec length of data index
    file_id = 0
    id_len = len(str( int(height*width/(stride[0]*stride[1])) ))
    
    for row in range(0, height-window_size[0], stride[0]):
        for col in range(0, width-window_size[1], stride[1]):
            data = bg[row:row+window_size[0], col:col+window_size[1], :]
            file_name = args.output + '/img' + str(file_id).zfill(id_len) + '.png'
            cv.imwrite(file_name, data)
            file_id += 1

        print('Processing row ' + str(row))
    
if __name__ == '__main__':
    main()