import sys, cv2
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python main.py [Image_path] [Task]")
    print("Tasks: upside-down, right-side-left, diagonally-mirrored, 45-clockwise, shrink, binarize")
    exit()

img = np.array(cv2.imread(sys.argv[1], 0))
result = np.zeros(img.shape)

if sys.argv[2] == "upside-down":    
    for i in range(img.shape[0]):
        result[i] = img[img.shape[0] - i - 1]

elif sys.argv[2] == "right-side-left":
    for i in range(img.shape[1]):
        result[:, i] = img[:, img.shape[1] - i - 1]

elif sys.argv[2] == "diagonally-mirrored":
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = img[j, i]

elif sys.argv[2] == "45-clockwise":
    result_pre = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    result = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result_pre[i + j, i - j] = img[j, i]
    result[:, 0 : int(result.shape[1] / 2)] = result_pre[:, int(result.shape[1] / 2) : result.shape[1]]
    result[:, int(result.shape[1] / 2) : result.shape[1]] = result_pre[:, 0 : int(result.shape[1] / 2)]

    result_f = np.zeros(img.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result_f[int(i/2), int(j/2)] += result[i, j]
    result = result_f / 2
    

elif sys.argv[2] == "shrink":
    result = np.zeros((int(img.shape[0] / 2), int(img.shape[1] / 2)))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[int(i/2), int(j/2)] += img[i, j]
    result /= 4

elif sys.argv[2] == "binarize":
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = (255 if img[i, j] > 128 else 0)

else:
    print("Wrong TaskName.")
    exit()

cv2.imwrite(sys.argv[2] + ".bmp", result)