import cv2 as cv
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Generate pictures from videos.')
    parser.add_argument('--input_path', help='Input video file path.', default='./data/raw/')
    parser.add_argument('--input_name', help='Input video file name.')
    parser.add_argument('--picnum', help='Number of generated pictures', default=20)
    parser.add_argument('--output', help='Output folder path.', default='./data/training-data/pos')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    input_file = args.input_path + args.input_name
    
    vc = cv.VideoCapture(input_file)
    if vc.isOpened():
        print('Video opened!')
    else:
        print("Video open error!")
    
    num_frames = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
    interval = num_frames // (args.picnum + 1)
    pic_id = 0
    id_len = len(str(args.picnum))
    
    for frame_id in range(interval, num_frames-interval+1, interval):
        vc.set(cv.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = vc.read()
        if ret:
            filename = args.output + '/' + str.split(args.input_name, '.')[0]\
                       + '_' + str(pic_id).zfill(id_len) + '.png' 
            cv.imwrite(filename, frame)
            pic_id += 1
        else:
            print('(!)-- Video open error!')
            exit(0)
            

if __name__ == '__main__':
    main()