import sys, cv2
import numpy as np

# Octogonal 3-5-5-5-3 kernel (+2)
kernelO_x = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
kernelO_y = [1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3]

def dilation(img_o):
  img_t = np.zeros(img_o.shape, dtype=np.int32)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      if img_o[i, j] > 0:
        max_l = 0
        # Find Max
        for x, y in zip(kernelO_x, kernelO_y):
          if i+x-2 > -1 and i+x-2 < img_o.shape[0] and j+y-2 > -1 \
            and j+y-2 < img_o.shape[1]:
            max_l = max(max_l, img_o[i+x-2, j+y-2])
        # Propagate
        for x, y in zip(kernelO_x, kernelO_y):
          if i+x-2 > -1 and i+x-2 < img_o.shape[0] and j+y-2 > -1 \
            and j+y-2 < img_o.shape[1]:
            img_t[i+x-2, j+y-2] = max_l
  return img_t

def erosion(img_o):
  img_t = np.zeros(img_o.shape, dtype=np.int32)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      min_l = 256
      for x, y in zip(kernelO_x, kernelO_y):
        # Boundaries, confirm all > 0, and find the Min
        if i+x-2 < 0 or i+x-2 > img_o.shape[0]-1 or j+y-2 < 0 \
          or j+y-2 > img_o.shape[1]-1 or img_o[i+x-2, j+y-2] == 0:
          min_l = -1; break
        else: min_l = min(min_l, img_o[i+x-2, j+y-2])
      # Propagate
      if min_l != -1:
        for x, y in zip(kernelO_x, kernelO_y):
          img_t[i+x-2, j+y-2] = min_l
  return img_t

def main():
  # Read the input image
  img = np.array(cv2.imread(sys.argv[1], 0))

  # Morphological dilation
  cv2.imwrite("hw5_dilation.bmp", dilation(img))
  # Morphological erosion
  cv2.imwrite("hw5_erosion.bmp", erosion(img))
  # Opening
  cv2.imwrite("hw5_opening.bmp", dilation(erosion(img)))
  # Closing
  cv2.imwrite("hw5_closing.bmp", erosion(dilation(img)))

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()
  main()