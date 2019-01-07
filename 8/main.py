import sys, cv2
import numpy as np
import math, random
from hw5_main import dilation, erosion

# Read the input image (Globalize original lena)
img = np.array(cv2.imread(sys.argv[1], 0))

# Signal-to-noise ratio
def snr(img_t):
  VS = np.std(img)
  VN = np.std(img_t - img)
  return 20 * math.log((VS/VN), 10)

# Generate additive white Gaussian noise 
def gaussian(img_o, amplitude):
  img_t = img_o + (amplitude * np.random.normal(0, 1, img_o.shape))
  print("\tGaussian noise with amplitude %d: %.4f" % (amplitude, snr(img_t))) 
  return img_t

# Generate salt-and-pepper noise 
def saltpepper(img_o, threshold):
  img_t = np.array(img_o)
  for i in range(img_o.shape[0]):
    for j in range(img_o.shape[1]):
      value = random.uniform(0, 1)
      if value < threshold: img_t[i, j] = 0
      elif value > (1 - threshold): img_t[i, j] = 255
  print("\tsalt-and-pepper noise with threshold %.2f: %.4f" % (threshold, snr(img_t)))
  return img_t

# Run box(True)/median(False) filter (3X3, 5X5) on all noisy images
def filters(img_o, size, box = True):
  img_t = np.array(img_o)
  img_o_n = np.pad(img_o, pad_width=2, mode='constant', constant_values=0)
  for i in range(img_t.shape[0]):
    for j in range(img_t.shape[1]):
      if box: img_t[i, j] = np.mean(img_o_n[2+i-size : 2+i+size, 2+j-size : 2+j+size])
      else: img_t[i, j] = np.median(img_o_n[2+i-size : 2+i+size, 2+j-size : 2+j+size])
  if box: print("\tBox_filter %dx%d: %.4f" % (size*2+1, size*2+1, snr(img_t)))
  else: print("\tMedian_filter %dx%d: %.4f" % (size*2+1, size*2+1, snr(img_t)))
  return img_t

# Opening then closing(True),  Closing then opening(False)
def openclose(img_o, order = True):
  if order: 
    img_t = erosion(dilation(dilation(erosion(img_o))))
    print("\tOpen then close: %.4f" % snr(img_t))
  else: 
    img_t = dilation(erosion(erosion(dilation(img_o))))
    print("\tClose then open: %.4f" % snr(img_t))
  return img_t

def main():
  noises = ['gaussian_10', 'gaussian_30', 'salt-and-pepper_01', 'salt-and-pepper_005']
  noise_pics = []
  
  print("PSNR: ")
  # Add noise
  noise_pics.append(gaussian(img, 10))
  noise_pics.append(gaussian(img, 30))
  noise_pics.append(saltpepper(img, 0.1))
  noise_pics.append(saltpepper(img, 0.05))
  
  # Removal
  for i in range(len(noise_pics)):
    print(noises[i] + ':')
    cv2.imwrite("hw8_noise_%s.bmp" % noises[i], noise_pics[i])
    cv2.imwrite("hw8_filter_box_3x3_%s.bmp" % noises[i], filters(noise_pics[i], 1, True))
    cv2.imwrite("hw8_filter_box_5x5_%s.bmp" % noises[i], filters(noise_pics[i], 2, True))
    cv2.imwrite("hw8_median_box_3x3_%s.bmp" % noises[i], filters(noise_pics[i], 1, False))
    cv2.imwrite("hw8_median_box_5x5_%s.bmp" % noises[i], filters(noise_pics[i], 2, False))
    cv2.imwrite("hw8_open_then_close_%s.bmp" % noises[i], openclose(noise_pics[i], True))
    cv2.imwrite("hw8_close_then_open_%s.bmp" % noises[i], openclose(noise_pics[i], False))
    

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()
  main()