import sys, cv2
import numpy as np

# Downsample to 64x64 and binarize
def downsample_binary(img_o):
    img_t = np.zeros((66, 66), dtype=np.int32)
    for i in range(64):
        for j in range(64):
            img_t[i+1, j+1] = (0 if img_o[8 * i, 8 * j] < 128 else 1)
    return img_t

# h operation
def _h(b, c, d, e):
    if b == c and (d != b or e != b): return 'q'
    elif b == c and (d == b and e == b): return 'r'
    else: return 's'

# f operation
def _f(neighbors):
    if(neighbors.count('r') == 4): return 5
    else: return neighbors.count('q')

def main():
    # Read the input image
    img = np.array(cv2.imread(sys.argv[1], 0))
    # d For code readability, 66x66 avoiding out of bound
    d = img_down = downsample_binary(img)
    
    # Yokoi f(a1, a2, a3, a4)
    result = [[' ' for x in range(64)] for y in range(64)]
    for i in range(1, 65):
        for j in range(1, 65):
            if img_down[i][j] != 0:
                result[j-1][i-1] = str(_f([_h(d[i, j], d[i, j+1], d[i-1, j+1], d[i-1, j]), 
                                    _h(d[i, j], d[i-1, j], d[i-1, j-1], d[i, j-1]), 
                                    _h(d[i, j], d[i, j-1], d[i+1, j-1], d[i+1, j]), 
                                    _h(d[i, j], d[i+1, j], d[i+1, j+1], d[i, j+1])]))
    # Output the result
    with open('hw6_result.txt', 'w') as fp:
        for i in range(64):
            for j in range(64):
                print(result[j][i], end='', file=fp)
            print(file=fp)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py [Image_path]")
        exit()
    main()