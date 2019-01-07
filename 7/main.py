import sys, cv2
import numpy as np
from skimage.morphology import skeletonize

# Downsample to binarized 64x64
def downsample_binary(img_o):
    # 66x66 for boundaries handling
    img_t = np.zeros((68, 68), dtype=np.int32)
    for i in range(64):
        for j in range(64):
            img_t[i+2, j+2] = (0 if img_o[8 * i, 8 * j] < 128 else 1)
    return img_t

# 4-connected
neighbor = [(-1, 0), (0, -1), (1, 0), (0, 1)]

def padding(img_o):
    img_t = np.array(img_o)
    for i in range(64):
        img_t[0, i+2] = img_t[1, i+2] = -1
        img_t[67, i+2] = img_t[66, i+2] = -1
        img_t[i+2, 0] = img_t[i+2, 1] = -1
        img_t[i+2, 67] = img_t[i+2, 66] = -1
    return img_t

# Mark-interior/border-pixel operator ((i, b) = (1, 2); 4-connected) 
def operator_ib(img_o):
    img_t = np.zeros((68, 68), dtype=np.int32)
    for i in range(2, 66):
        for j in range(2, 66):
            if img_o[i, j] == 1:
                img_t[i, j] = 1
                for (x, y) in neighbor:
                    if img_o[i+x, j+y] != 1: img_t[i, j] = 2; break
    return img_t

# Pair relationship operator ((p = 3); 4-connected)
def operator_pr (img_o):
    img_t = np.zeros((68, 68), dtype=np.int32)
    for i in range(2, 66):
        for j in range(2, 66):
            if img_o[i, j] == 2:
                for (x, y) in neighbor:
                    if img_o[i+x, j+y] == 1: img_t[i, j] = 1; break
    return img_t

def _h(b, c, d, e):
    if b == c and (b != d or b != e): return 1
    return 0

def yokoi(img_o, i, j):
    _f = [_h(img_o[i, j], img_o[i, j+1], img_o[i-1, j+1], img_o[i-1, j]),
            _h(img_o[i, j], img_o[i-1, j], img_o[i-1, j-1], img_o[i, j-1]),
            _h(img_o[i, j], img_o[i, j-1], img_o[i+1, j-1], img_o[i+1, j]),
            _h(img_o[i, j], img_o[i+1, j], img_o[i+1, j+1], img_o[i, j+1])]
    if np.array(_f).sum() == 1: return True
    return False

def main():
    # Read the input image
    img = np.array(cv2.imread(sys.argv[1], 0))
    # Downsample to binarized 64x64
    img = downsample_binary(img)
    cv2.imwrite("jizz.bmp", img * 255)

    count = 0
    while 1:
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
        print(count)
        count += 1
        img_ref = np.array(img)
        img_ib = operator_ib(img)
        img_pr = operator_pr(img_ib)
        for i in range(2, 66):
            for j in range(2, 66):
                if yokoi(img, i, j) and img_ib[i, j] == 2 and img_pr[i, j] == 1:
                    img[i, j] = 0

        #cv2.imwrite(str(count) + "hw7_result.bmp", img[2:66, 2:66] * 255)
        if np.array_equal(img, img_ref): break

    cv2.imwrite("hw7_result.bmp", img[2:66, 2:66] * 255)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py [Image_path]")
        exit()
    main()
