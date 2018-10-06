import sys, cv2, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python main.py [Image_path] [Task]")
    print("Tasks: binary, histogram, connected_components")
    exit()

img = np.array(cv2.imread(sys.argv[1], 0))
b_img = np.zeros(img.shape, dtype=np.int32)
c_all = []

def binary():
    c_label = 0
    for i in range(img.shape[0]):
        # ROW
        for j in range(img.shape[1]):
            b_img[i, j] = (0 if img[i, j] < 128 else 1)
        
        # Connect components
        k, c_row = 0, []
        while k < img.shape[1]:
            if b_img[i, k] == 1:
                c_dot = [c_label, k]
                while k < img.shape[1] and b_img[i, k] == 1: k += 1
                c_dot.append(k-1)
                # Update the dot in row
                c_row.append(c_dot)
                c_label += 1
            else: k += 1
        # Update the row in all
        c_all.append(c_row)

def overlap(a, b):
    if a[0] == b[0]: return False
    elif a[1] < b[1] and a[2] < b[1]: return False
    elif a[1] > b[2] and a[2] > b[2]: return False
    else : return True

if sys.argv[2] == "binary":
    binary()
    # Output
    cv2.imwrite("hw2_binary.bmp", b_img * 255)

elif sys.argv[2] == "histogram":
    result, x = np.zeros(256, dtype=np.int32), list(range(256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): result[img[i, j]] += 1
    plt.rcParams['axes.facecolor'] = 'black'
    plt.bar(x, result, color='white', edgecolor='white')
    plt.savefig("hw2_histogram.jpg")

elif sys.argv[2] == "connected_components":
    binary()
    update = True
    while update:
        update = False
        # Top-left to Bottom-right
        for i in range(1, len(c_all)):
            for j in range(len(c_all[i])):
                for k in range(len(c_all[i-1])):
                    if overlap(c_all[i][j], c_all[i-1][k]):
                        c_all[i][j][0] = c_all[i-1][k][0] = min(c_all[i][j][0], c_all[i-1][k][0])
                        update = True

        # Bottom-right to Top-left
        for i in reversed(range(1, len(c_all))):
            for j in reversed(range(len(c_all[i]))):
                for k in reversed(range(len(c_all[i-1]))):
                    if overlap(c_all[i][j], c_all[i-1][k]):
                        c_all[i][j][0] = c_all[i-1][k][0] = min(c_all[i][j][0], c_all[i-1][k][0])
                        update = True
    
    # Draw the bounding boxes
    random.seed(100)
    r = np.zeros(260000, dtype=np.int32)
    g = np.zeros(260000, dtype=np.int32)
    b = np.zeros(260000, dtype=np.int32)
    c_labels = np.zeros((260000, 5), dtype=np.int32)
    for i in range(260000):
        c_labels[i, 1] = c_labels[i, 3] = 512
        r[i] = random.randint(80, 255)
        g[i] = random.randint(80, 255)
        b[i] = random.randint(80, 255)

    c_img = np.zeros((512, 512, 3), dtype=np.int32)
    for i in range(len(c_all)):
        for j in range(len(c_all[i])):
            for k in range(c_all[i][j][1], c_all[i][j][2]):
                label = c_all[i][j][0]
                c_labels[label, 0] += 1
                # Barriers
                if i < c_labels[label, 1]: c_labels[label, 1] = i
                if i > c_labels[label, 2]: c_labels[label, 2] = i
                if k < c_labels[label, 3]: c_labels[label, 3] = k
                if k > c_labels[label, 4]: c_labels[label, 4] = k
                # Color image
                c_img[i, k, 0] = r[c_all[i][j][0]]
                c_img[i, k, 1] = g[c_all[i][j][0]]
                c_img[i, k, 2] = b[c_all[i][j][0]]

    for i in range(260000):
        if c_labels[i, 0] > 500:
            # Boxes
            cv2.rectangle(c_img, (c_labels[i, 3], c_labels[i, 1]), 
                (c_labels[i, 4], c_labels[i, 2]), (0, 0, 200), thickness=3)
            # Center
            cv2.drawMarker(c_img, (int(0.5*(c_labels[i, 3] + c_labels[i, 4])), 
                int(0.5*(c_labels[i, 1] + c_labels[i, 2]))), (0, 0, 255), thickness=2)
    
    cv2.imwrite("hw2_boundboxes.bmp", c_img)

else:
    print("Wrong TaskName.")
    exit()
