import sys, cv2
import numpy as np

# Octogonal 3-5-5-5-3 kernel (+2)
kernelO_x = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
kernelO_y = [1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3]

def binary(img_o):
  img_t = np.zeros(img_o.shape, dtype=np.int32)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      img_t[i, j] = (0 if img_o[i, j] < 128 else 1)
  return img_t

def dilation(img_o):
  img_t = np.zeros(img_o.shape, dtype=np.int32)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      if img_o[i, j] == 1:
        for x, y in zip(kernelO_x, kernelO_y):
          # Boundaries
          if i+x-2 > -1 and i+x-2 < img_o.shape[0] and j+y-2 > -1 \
            and j+y-2 < img_o.shape[1]:
            img_t[i+x-2, j+y-2] = 1
  return img_t

def erosion(img_o):
  img_t = np.zeros(img_o.shape, dtype=np.int32)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      img_t[i, j] = 1
      for x, y in zip(kernelO_x, kernelO_y):
        # Boundaries and confirm 1
        if i+x-2 < 0 or i+x-2 > img_o.shape[0]-1 or j+y-2 < 0 \
          or j+y-2 > img_o.shape[1]-1 or img_o[i+x-2, j+y-2] != 1:
          img_t[i, j] = 0; break
  return img_t

def hit_and_miss(img_o):
  global kernelO_x; global kernelO_y
  # J kernel (A - J)
  kernelO_x = [1, 1, 2]; kernelO_y = [2, 3, 3]
  img_J = erosion(img_o)
  # K kernel (Ac - K)
  kernelO_x = [0, 0, 1]; kernelO_y = [3, 4, 4]
  img_K = erosion(np.ones(img_o.shape, dtype=np.int32) - img_o)
  # 1 only if both pixel are 2
  return (img_J + img_K) // 2

def main():
  # Read the input image
  img = np.array(cv2.imread(sys.argv[1], 0))

  # Binarize first
  img_binary = binary(img)
  cv2.imwrite("hw4_binary.bmp", img_binary*255)

  # Binary morphological dilation
  cv2.imwrite("hw4_dilation.bmp", dilation(img_binary)*255)
  # Binary morphological erosion
  cv2.imwrite("hw4_erosion.bmp", erosion(img_binary)*255)
  # Opening
  cv2.imwrite("hw4_opening.bmp", dilation(erosion(img_binary))*255)
  # Closing
  cv2.imwrite("hw4_closing.bmp", erosion(dilation(img_binary))*255)
  # Hit-and-miss transform 
  cv2.imwrite("hw4_hit-and-miss.bmp", hit_and_miss(img_binary)*255)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()
  main()