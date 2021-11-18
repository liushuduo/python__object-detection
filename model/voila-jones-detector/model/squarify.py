import argparse
import cv2 as cv

def get_parser():
    parser = argparse.ArgumentParser(description='Unify bounding box size after opencv_annotation.')
    parser.add_argument('--file', help='Filename for annotation file.')
    parser.add_argument('--width', help='Resized bounding box width in pixels', default=46)
    parser.add_argument('--compfile', help='Demo file to compare bounding box before and after processing', default=None)
    return parser
    
    
def squarify(file, square_width):
    file_name = file.strip('.txt')
    file_name = file_name + "_sqr.txt"
    fn = open(file_name, "w")
    
    with open(file) as f:
        data = f.readlines()

        for line in data:
            d = line.split()
            num_d = int(d[1])
            new_data = d[0:2]
            
            for ind in range(num_d):
                x, y = int(d[2+ind*4]), int(d[3+ind*4])
                width, height = int(d[4+ind*4]), int(d[5+ind*4])
                center_x = x + width // 2
                center_y = y + height // 2
                
                new_x = center_x - square_width//2
                new_y = center_y - square_width//2
                
                new_data = new_data + [str(new_x)] + [str(new_y)] + [str(square_width)] + [str(square_width)]
                
            fn.write(" ".join(new_data) + "\n")
            
            
def compare_changes(file, img_file):
    with open(file) as f:
        before = f.readline().split(' ')
    
    with open(file.split('.')[0]+'_sqr.txt') as f:
        after = f.readline().split(' ')
    
    img = cv.imread(img_file)
    
    num_box = int(before[1])
    
    for ind in range(num_box):
        cv.rectangle(img, \
            (int(before[ind*4+2]), int(before[ind*4+3])), \
            (int(before[ind*4+2])+int(before[ind*4+4]), int(before[ind*4+3])+int(before[ind*4+5]) ), (255, 0, 0), 3)

        cv.rectangle(img, \
            (int(after[ind*4+2]), int(after[ind*4+3])), \
            (int(after[ind*4+2])+int(after[ind*4+4]), int(after[ind*4+3])+int(after[ind*4+5]) ), (0, 0, 255), 3)
    
    cv.imwrite("processTest.png", img)

    
def main():
    parser = get_parser()
    args = parser.parse_args()
    file = args.file
    square_width = args.width
    squarify(file=file, square_width=square_width)
    if args.compfile:
        compare_changes(file, args.compfile)


if __name__ == "__main__":
    main()