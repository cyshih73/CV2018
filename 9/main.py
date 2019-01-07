import sys, cv2
import numpy as np

shape = [0, 0]

def robert(img_o, thres):
  print("Robert's operator, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      r = (-img_o[i+2, j+2] + img_o[i+3, j+3]) ** 2
      r += (-img_o[i+2, j+3] + img_o[i+3, j+2]) ** 2
      img_t[i, j] = 0 if r ** 0.5 > thres else 1
  return img_t

def prewitt(img_o, thres):
  print("Prewitt's Edge Detector, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      p = (-np.sum(img_o[i+2, j+2:j+5]) + np.sum(img_o[i+4, j+2:j+5])) ** 2
      p += (-np.sum(img_o[i+2:i+5, j+2]) + np.sum(img_o[i+2:i+5, j+4])) ** 2
      img_t[i, j] = 0 if p ** 0.5 > thres else 1
  return img_t

def sobel(img_o, thres):
  print("Sobel's Edge Detector, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      s = (-np.sum(img_o[i+2, j+2:j+5]) - img_o[i+2, j+3] + 
          np.sum(img_o[i+4, j+2:j+5]) + img_o[i+4, j+3]) ** 2
      s += (-np.sum(img_o[i+2:i+5, j+2]) - img_o[i+3, j+2] + 
          np.sum(img_o[i+2:i+5, j+4]) + img_o[i+3, j+4]) ** 2
      img_t[i, j] = 0 if s ** 0.5 > thres else 1
  return img_t

def FandC(img_o, thres):
  print("Frei and Chen's Gradient Operator, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      f = (img_o[i+4, j+2] - img_o[i+2, j+2] + (2**0.5)*(img_o[i+4, j+3] - 
            img_o[i+2, j+3]) + img_o[i+4, j+4] - img_o[i+2, j+4]) ** 2
      f += (img_o[i+2, j+4] - img_o[i+2, j+2] + (2**0.5)*(img_o[i+3, j+4] - 
            img_o[i+3, j+2]) + img_o[i+4, j+4] - img_o[i+4, j+2]) ** 2
      img_t[i, j] = 0 if f ** 0.5 > thres else 1
  return img_t

def Kirsch(img_o, thres):
  print("Kirsch's Compass Operator, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      k = np.zeros(8, dtype=np.int16)
      k[0] = -3*(np.sum(img_o[i+1:i+4, j+1]) + img_o[i+1, j+2] + img_o[i+3, j+2]) + \
              5*np.sum(img_o[i+1:i+4, j+3])
      k[1] = -3*(np.sum(img_o[i+1:i+4, j+1]) + img_o[i+3, j+2] + img_o[i+3, j+3]) + \
              5*(img_o[i+1, j+2] + img_o[i+1, j+3] + img_o[i+2, j+3])
      k[2] = -3*(np.sum(img_o[i+3, j+1:j+4]) + img_o[i+2, j+1] + img_o[i+2, j+3]) + \
              5*np.sum(img_o[i+1, j+1:j+4])
      k[3] = -3*(np.sum(img_o[i+1:i+4, j+3]) + img_o[i+3, j+2] + img_o[i+3, j+1]) + \
              5*(img_o[i+1, j+1] + img_o[i+2, j+1] + img_o[i+1, j+2])
      k[4] = -3*(np.sum(img_o[i+1:i+4, j+3]) + img_o[i+1, j+2] + img_o[i+3, j+2]) + \
              5*np.sum(img_o[i+1:i+4, j+1])
      k[5] = -3*(np.sum(img_o[i+1:i+4, j+3]) + img_o[i+1, j+1] + img_o[i+1, j+2]) + \
              5*(img_o[i+2, j+1] + img_o[i+3, j+1] + img_o[i+3, j+2])
      k[6] = -3*(np.sum(img_o[i+1, j+1:j+4]) + img_o[i+2, j+1] + img_o[i+2, j+3]) + \
              5*np.sum(img_o[i+3, j+1:j+4])
      k[7] = -3*(np.sum(img_o[i+1:i+4, j+1]) + img_o[i+1, j+2] + img_o[i+1, j+3]) + \
              5*(img_o[i+2, j+3] + img_o[i+3, j+3] + img_o[i+3, j+2])
      img_t[i, j] = 0 if np.max(k) > thres else 1
  return img_t

def robinson(img_o, thres):
  print("Robinson's Compass Operator, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      r = np.zeros(8, dtype=np.int16)
      r[0] = -1*(img_o[i+1, j+1] + img_o[i+3, j+1]) + img_o[i+1, j+3] + img_o[i+3, j+3] + 2*img_o[i+2, j+3] - 2*img_o[i+2, j+1]
      r[1] = -1*(img_o[i+2, j+1] + img_o[i+3, j+2]) + img_o[i+1, j+2] + img_o[i+2, j+3] + 2*img_o[i+1, j+3] - 2*img_o[i+3, j+1]
      r[2] = -1*(img_o[i+3, j+1] + img_o[i+3, j+3]) + img_o[i+1, j+1] + img_o[i+1, j+3] + 2*img_o[i+1, j+2] - 2*img_o[i+3, j+2]
      r[3] = -1*(img_o[i+3, j+2] + img_o[i+2, j+3]) + img_o[i+2, j+1] + img_o[i+1, j+2] + 2*img_o[i+1, j+1] - 2*img_o[i+3, j+3]
      r[4], r[5], r[6], r[7] = -r[0], -r[1], -r[2], -r[3] 
      img_t[i, j] = 0 if np.max(r) > thres else 1
  return img_t

def nevatia(img_o, thres):
  print("Nevatia-Babu 5x5 Operator, threshold = %d" % thres)
  img_t = np.zeros(shape, dtype=np.int16)
  for i in range(shape[0]):
    for j in range(shape[1]):
      n = np.zeros(6, dtype=np.int16)
      n[0] = 100*(np.sum(img_o[i:i+2, j:j+5])) + -100*(np.sum(img_o[i+3:i+5, j:j+5]))   # 0

      n[1] = 100*(np.sum(img_o[i, j:j+5]) + np.sum(img_o[i+1, j:j+3]) + img_o[i+2, j]) \
            + 78*img_o[i+1, j+3] - 32*img_o[i+1, j+4] + 92*img_o[i+2, j+1] - 92*img_o[i+2, j+3] + 32*img_o[i+3, j] - 78*img_o[i+3, j+1] \
            - 100*(np.sum(img_o[i+4, j:j+5]) + np.sum(img_o[i+3, j+2:j+5]) + img_o[i+2, j+4])   # 30

      n[2] = 100*(np.sum(img_o[i:i+5, j]) + np.sum(img_o[i:i+3, j+1]) + img_o[i, j+2]) \
            + 78*img_o[i+3, j+1] - 32*img_o[i+4, j+1] + 92*img_o[i+1, j+2] - 92*img_o[i+3, j+2] + 32*img_o[i, j+3] - 78*img_o[i+1, j+3] \
            - 100*(np.sum(img_o[i:i+5, j+4]) + np.sum(img_o[i+2:i+5, j+3]) + img_o[i+4, j+2])   # 60

      n[3] = 100*(np.sum(img_o[i:i+5, j+3:j+5])) + -100*(np.sum(img_o[i:i+5, j:j+2]))   # -90

      n[4] = 100*(np.sum(img_o[i:i+5, j+4]) + np.sum(img_o[i:i+3, j+3]) + img_o[i, j+2]) \
            + 78*img_o[i+3, j+3] - 32*img_o[i+4, j+3] + 92*img_o[i+1, j+2] - 92*img_o[i+3, j+2] + 32*img_o[i, j+1] - 78*img_o[i+1, j+1] \
            - 100*(np.sum(img_o[i:i+5, j]) + np.sum(img_o[i+2:i+5, j+1]) + img_o[i+4, j+2])   # -60

      n[5] = 100*(np.sum(img_o[i, j:j+5]) + np.sum(img_o[i+1, j+2:j+5]) + img_o[i+2, j+4]) \
            + 78*img_o[i+1, j+1] - 32*img_o[i+1, j] - 92*img_o[i+2, j+1] + 92*img_o[i+2, j+3] + 32*img_o[i+3, j+4] - 78*img_o[i+3, j+3] \
            - 100*(np.sum(img_o[i+4, j:j+5]) + np.sum(img_o[i+3, j:j+3]) + img_o[i+2, j])   # -30

      img_t[i, j] = 0 if np.max(n) > thres else 1
  return img_t


def main():
  # Read the input image
  global shape
  img = np.array(cv2.imread(sys.argv[1], 0), dtype=np.int16)
  shape = img.shape
  img = np.pad(img, pad_width=2, mode='constant', constant_values=0)

  cv2.imwrite("hw9_Robert.bmp", robert(img, 12)*255)
  cv2.imwrite("hw9_Prewitt.bmp", prewitt(img, 24)*255)
  cv2.imwrite("hw9_Sobel.bmp", sobel(img, 38)*255)
  cv2.imwrite("hw9_Frei_and_Chen.bmp", FandC(img, 30)*255)
  cv2.imwrite("hw9_Kirsch.bmp", Kirsch(img, 135)*255)
  cv2.imwrite("hw9_Robinson.bmp", robinson(img, 43)*255)
  cv2.imwrite("hw9_Nevatia_Babu.bmp", nevatia(img, 12500)*255)
  

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()
  main()