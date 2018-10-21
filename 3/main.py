import sys, cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python main.py [Image_path]")
    exit()

# Histogram for original, darkened and result imgs
his = [np.zeros(256), np.zeros(256), np.zeros(256)]
# Cumulative Distribution Function
cdf = np.zeros(256)

img = np.array(cv2.imread(sys.argv[1], 0))
dark_img = img.copy()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # Darken img by 2/3
        dark_img[i, j] = 2 * (img[i, j] / 3)
        his[0][img[i, j]] += 1
        his[1][dark_img[i, j]] += 1

count, min_n = 0, 0
for i in range(256):
    if min_n == 0 and his[1][i] != 0: min_n = i
    count += his[1][i]
    cdf[i] = count

result = img.copy()
n_pixels = float((img.shape[0] * img.shape[1]) - cdf[min_n])
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        y = round(255.0 * float(cdf[dark_img[i , j]] - cdf[min_n]) / n_pixels)
        result[i , j] = y
        his[2][result[i, j]] += 1

# Output the image
cv2.imwrite("hw3_dark.bmp", dark_img)
cv2.imwrite("hw3_result.bmp", result)

# Plot the histograms
plt.figure()
plt.rcParams['axes.facecolor'] = 'black'
plt.bar(range(0 , 256) , his[0], color='white', edgecolor='white')
plt.savefig("hw3_histo_Original.jpg")
plt.figure()
plt.bar(range(0 , 256) , his[1], color='white', edgecolor='white')
plt.savefig("hw3_histo_Dark.jpg")
plt.figure()
plt.bar(range(0 , 256) , his[2], color='white', edgecolor='white')
plt.savefig("hw3_histo_Result.jpg")