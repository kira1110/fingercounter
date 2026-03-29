import cv2
import numpy as np
import os

data = []
labels = []

dataset = "finger_dataset"

for label in os.listdir(dataset):

    folder = os.path.join(dataset,label)

    for img in os.listdir(folder):

        path = os.path.join(folder,img)

        image = cv2.imread(path)
        image = cv2.resize(image,(64,64))

        data.append(image)
        labels.append(int(label))

data = np.array(data)/255.0
labels = np.array(labels)

np.save("X.npy",data)
np.save("y.npy",labels)

print("Dataset saved")