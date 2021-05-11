# Importing needed libraries

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Canny Edge dedection
def Canny_edge(img):
    # Canny Edge
    canny_edges = cv2.Canny(img, 8, 8)
    return canny_edges


# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


# Read paths for images in different folders
path_covid = 'C:/Users/Alihan/Desktop/sample/COVID'
path_pneumonia = 'C:/Users/Alihan/Desktop/sample/Viral Pneumonia'
path_normal = 'C:/Users/Alihan/Desktop/sample/NORMAL'
labels = []
images = []
filtered_images = []

# assigning relevant data with appropriate arrays
for i in os.listdir(path_covid):
    img = cv2.imread(os.path.join(path_covid, i))
    img = cv2.resize(img, (8, 8))
    label = i.split(" ")[0]
    labels.append(label)
    images.append(img)

for i in os.listdir(path_pneumonia):
    img = cv2.imread(os.path.join(path_pneumonia, i))
    img = cv2.resize(img, (8, 8))
    label = i.split(" ")[0]
    labels.append(label)
    images.append(img)

for i in os.listdir(path_normal):
    img = cv2.imread(os.path.join(path_normal, i))
    img = cv2.resize(img, (8, 8))
    label = i.split(" ")[0]
    labels.append(label)
    images.append(img)

training = np.array(images)
labels = np.array(labels)

# Extracting Features
for i in range(len(training)):  # For loop to process all train data
    img = Gabor_process(training[i])  # Gabor process first
    img = Canny_edge(img)  # Canny_edge filter after that
    filtered_images.append(img)  # appending filtered images in filtered_images array

filtered_images = np.array(filtered_images)

kf = KFold(n_splits=7, random_state=None, shuffle=True)  # K-fold process here

X_train, X_test, y_train, y_test = 0, 0, 0, 0
for train_index, test_index in kf.split(
        filtered_images):  # For loop for assigning train and test data by k-fold process
    X_train, X_test = filtered_images[train_index], filtered_images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


# k-nearest neighbour algorithm

def knn(train, test, labelArr, k):
    prediction = []
    for i in range(len(test)):  # For loops to get euclidean distance between all train and test sets
        distances = []
        for j in range(len(train)):
            euclidean_dis = np.linalg.norm(train[j] - test[i])
            distances.append(euclidean_dis)
        sorted_labels = []
        for a, b in sorted(zip(distances, labelArr)):  # Sorting labels due to sorted distances
            sorted_labels.append(b)
        prediction.append(knn_Prediction(sorted_labels, k))  # Prediction array appends with predicted data
        np.array(prediction)

    acc = get_AccuracyKnn(prediction, y_test)  # Accruacy function returns mean accuracy

    return acc


def knn_Prediction(sortedArr, k):  # This function returns most repeated value of array
    return Counter(np.array(sortedArr[:k])).most_common(1)[0][0]


# Weighted k-nearest neighbour algorithm

def wknn(train, test, labelArr, k):
    predictionArr = []
    for i in range(len(test)):  # For loops to get euclidean distance between all train and test sets
        distances = []
        for j in range(len(train)):
            euclidean_dis = np.linalg.norm(train[j] - test[i])
            distances.append(euclidean_dis)
        sorted_labels = []
        for dist, labelVar in sorted(zip(distances, labelArr)):  # Sorting labels due to sorted distances
            sorted_labels.append(labelVar)
        distances = sorted(distances)

        predictionArr.append(
            wknn_Prediction(sorted_labels, distances, k))  # Prediction array appends with predicted data

    np.array(predictionArr)
    acc = get_AccuracyKnn(predictionArr, y_test)  # Accuracy function returns mean accuracy

    return acc


# wknn Prediction function

def wknn_Prediction(sorted_labels, distances, k):
    for p in range(k):  # For loop to find frequency
        freq_cov, freq_pneu, freq_norm = 0, 0, 0
        if sorted_labels[p] == "COVID":
            if distances[p] != 0:  # If distance zero, ignore it
                freq_cov += 1 / distances[p]

        elif sorted_labels[p] == "Viral":
            if distances[p] != 0:
                freq_pneu += 1 / distances[p]

        elif sorted_labels[p] == "NORMAL":
            if distances[p] != 0:
                freq_norm += 1 / distances[p]

    if max(freq_cov, freq_norm, freq_pneu) == freq_cov:  # Find maximum frequency to get label
        prediction = "COVID"
    elif max(freq_cov, freq_norm, freq_pneu) == freq_pneu:
        prediction = "Viral"
    elif max(freq_cov, freq_norm, freq_pneu) == freq_norm:
        prediction = "NORMAL"
    else:
        prediction = sorted_labels[0]

    return prediction


# Function that calculate accuracy

def get_AccuracyKnn(predicted, test):
    correct, total = 0, 0
    acc = []
    for i in range(np.size(predicted)):
        if test[i] == predicted[i]:
            correct = correct + 1
        total = total + 1
        acc.append((correct / total) * 100)

    return round(sum(acc) / len(acc))  # returns accuracy in percentage


k = 6
print("K- value: ", k)
print("MEAN ACCURACY FOR KNN: " + str(knn(X_train, X_test, y_train, k)))
print("MEAN ACCURACY FOR WEIGHTED KNN: " + str(wknn(X_train, X_test, y_train, k)))
print("--------------------------")
