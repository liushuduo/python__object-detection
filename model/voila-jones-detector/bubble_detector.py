import cv2 as cv
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Detect bubbles using Voila-Jones detector')
    parser.add_argument('--file', help='File name for picture(png/jpeg) or video (mp4)', default='voila-jones-detector/data/pos/segment01.mp4_20211116_151559.161.jpg')
    parser.add_argument('--cascade', help='Path to cascade', default='voila-jones-detector/data/bubble-classifier/cascade.xml')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    file_name = args.file 
    bubble_cascade_name = args.cascade
    
    bubble_cascade = cv.CascadeClassifier(bubble_cascade_name)
    
    img = cv.imread(file_name)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    bubbles = bubble_cascade.detectMultiScale(img_gray)
    print(bubbles)
    for (x, y, w, h) in bubbles:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
            
                