import gzip
import sys
import pandas as pd
import numpy as np
import time
from Model import Model
from Visualization import PlotConfusionMatrix, convert, ShowExamples

image_size = 28
from sklearn.metrics import confusion_matrix

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def read_data():
    files_images = []
    files_images.append('train-images-idx3-ubyte.gz')
    files_images.append('t10k-images-idx3-ubyte.gz')
    files_labels = []
    files_labels.append('train-labels-idx1-ubyte.gz')
    files_labels.append('t10k-labels-idx1-ubyte.gz')
    images = []
    list_labels = []
    for file_images, file_labels in zip(files_images, files_labels):
        f = gzip.open(file_images, 'r')
        f.read(16)
        buf = f.read()
        f.close()
        num_images = sys.getsizeof(buf) // (image_size * image_size)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X = data.reshape(num_images, 1, image_size, image_size) / 255
        images.append(X)

        f = gzip.open(file_labels, 'r')
        f.read(8)
        buf = f.read()
        f.close()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels = labels.reshape(num_images)
        list_labels.append(labels)
    return images[0], list_labels[0], images[1], list_labels[1]


def accuary(model, x, y, name, size_batch = 1024):
    loss = []
    dummies_y = pd.get_dummies(y)
    con_matrix=np.zeros((len(digits), len(digits)))
    for fnum in range(0, len(dummies_y), size_batch):
        split_x=x[fnum:fnum + size_batch]
        split_y=y[fnum:fnum + size_batch]
        temp_loss = model.Loss(split_x, dummies_y[fnum:fnum + size_batch])
        loss.append(temp_loss.detach().numpy())
        y_pred = convert(model, split_x)
        cm = confusion_matrix(split_y, y_pred)
        con_matrix+=cm
    correct = np.trace(con_matrix)
    loss = np.array(loss)
    mean_loss = loss.mean()
    print('%s: Accuracy %d/%d (%.0f%%), Loss %.3f;' % (name,
                                                       correct, len(y), 100. * correct / len(y), mean_loss))
    return


def train_loop(model, train_x, train_y, test_x, test_y, max_epochs=100):
    dummies_train_y = pd.get_dummies(train_y)
    for epoch in range(max_epochs + 1):
        start_time = time.time()
        size_batch = 1024
        for fnum in range(0, len(dummies_train_y), size_batch):
            model.Train(train_x[fnum:fnum + size_batch], dummies_train_y[fnum:fnum + size_batch])
        end_time = time.time()
        if epoch % 5 == 0:
            print('\nEpoch %d' % (epoch))
            print('Time of one epoch %.2f' % (end_time - start_time))
            accuary(model, train_x, train_y, 'Train', size_batch)
            accuary(model, test_x, test_y, 'Test', size_batch)
    PlotConfusionMatrix(model, train_x, train_y, test_x, test_y)
    return


model = Model('Handwritten Digits')
train_x, train_labels, test_x, test_labels = read_data()
train_loop(model, train_x, train_labels, test_x, test_labels)
ShowExamples(model, test_x, test_labels, 10)
