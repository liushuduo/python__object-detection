import cv2 as cv
import numpy as np
from skimage import feature


def predict_target(img, svm, window_size, stride):

    height, width = img.shape[0:2]

    # Sliding window
    for row in range(0, height-window_size[0], stride[0]):
        for col in range(0, width-window_size[1], stride[1]):
            
            # Region of interest.
            win_roi = img[row:row+window_size[0], col:col+window_size[1], :]

            # Compute HOG descriptor
            (ski_hog, ski_hog_image) = feature.hog(
                win_roi, orientations=9, 
                pixels_per_cell=(8, 8), cells_per_block=(2,2), 
                block_norm='L2-Hys', visualize=True, transform_sqrt=True
            )

            # ski_hog (n_feat,) --> hog_desc (1, n_feat)
            hog_desc = (np.reshape(ski_hog, [1, -1])).astype(np.float32)
            result = svm.predict(hog_desc)[1]

            if result[0][0] > 0:
                # mark detected target
                cv.rectangle(img, (col, row), np.array([col, row])+window_size[::-1], (0,0,255), 1)
    
    return img


def main():
    svm = cv.ml.SVM_load('./model/HOG-detector/HOG_svm.dat')
    img = cv.imread('./data/HOG_dataset/valid/segment01_03.png')
    result = predict_target(img, svm, [48, 48], [16, 16])
    cv.imwrite('./result/HOG_result.png', result)


if __name__ == '__main__':
    main()