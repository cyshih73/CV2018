import sys, cv2
import numpy as np

shape = [0, 0]

def Laplace(img_o, kernel, thres):
  pad = len(kernel) // 2
  img_t = np.ones(shape, dtype=np.int16)
  temp_val = np.zeros(img_o.shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      accum = np.sum(img_o[i+5-pad:i+6+pad, j+5-pad:j+6+pad] * kernel)
      temp_val[i+5, j+5] = 1 if (accum > thres) else -1 if (-accum > thres) else 0
  for i in range(shape[0]):
    for j in range(shape[1]):
      if temp_val[i+5, j+5] == 1 and np.isin(-1, temp_val[i+5-pad:i+6+pad, j+5-pad:j+6+pad]):
        img_t[i, j] = 0
  return img_t

def main():
  # Read the input image
  img = np.array(cv2.imread(sys.argv[1], 0), dtype=np.int16)
  global shape; shape = img.shape
  img = np.pad(img, pad_width=5, mode='constant', constant_values=0)
  
  Mask1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
  cv2.imwrite("hw10_LaplaceMask1.bmp", Laplace(img, Mask1, 15)*255)
  Mask2 = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
  cv2.imwrite("hw10_LaplaceMask2.bmp", Laplace(img, Mask2, 45)*255)
  Mini = [[2, -1, 2], [-1, -4, -1], [2, -1, 2]]
  cv2.imwrite("hw10_MinimumVarianceLaplacian.bmp", Laplace(img, Mini, 60)*255)
  LG = [ [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0] ,
           [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0] ,
           [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0] ,
           [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1] ,
           [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1] ,
           [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2] ,
           [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1] ,
           [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1] ,
           [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0] ,
           [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0] ,
           [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0] ]
  cv2.imwrite("hw10_LaplaceOfGaussian.bmp", Laplace(img, LG, 3000)*255)
  DG = [ [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1] ,
           [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3] ,
           [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4] ,
           [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6] ,
           [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7] ,
           [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8] ,
           [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7] ,
           [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6] ,
           [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4] ,
           [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3] ,
           [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1] ]
  cv2.imwrite("hw10_DifferenceOfGaussian.bmp", Laplace(img, DG, 1)*255)
  
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()
  main()