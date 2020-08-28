import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def convert(model, X):
    ProbYNon, ProbY=model.net(torch.FloatTensor(X))
    ProbY = ProbY.detach().numpy()
    ProbY = np.ndarray.tolist(ProbY)
    y_pred = []
    for y in ProbY:
         num = y.index(max(y))
         y_pred.append(digits[num])
    return y_pred

def ConMatrix(ax, model, X, Y,  name):
    size_batch=1024
    con_matrix = np.zeros((len(digits), len(digits)))
    for fnum in range(0, len(Y), size_batch):
        split_x=X[fnum:fnum + size_batch]
        split_y = Y[fnum:fnum + size_batch]
        y_pred = convert(model, split_x)
        cm=confusion_matrix(split_y, y_pred)
        con_matrix+=cm
    con_matrix=con_matrix/len(Y)
    disp = ConfusionMatrixDisplay(con_matrix, display_labels=digits)
    disp.plot(ax=ax, cmap='Blues', values_format='.3f')
    plt.sca(ax)
    plt.xticks(range(len(digits)), digits, fontsize=8)
    plt.yticks(range(len(digits)), digits, fontsize=8)
    disp.ax_.set_title(name)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    disp.ax_.set_ylabel('')
    return disp

def PlotConfusionMatrix(model, train_x, train_y, test_x, test_y):
    f, axes = plt.subplots(1, 2, figsize=(32, 16))
    f.suptitle('Confusion matrices', fontsize=16)
    D=ConMatrix(axes[0], model, train_x, train_y, 'Training')
    D.ax_.set_ylabel('True label')
    D=ConMatrix(axes[1], model, test_x, test_y, 'Testing')
    plt.subplots_adjust(left=0.15, wspace=0.4, hspace=0.01)
    f.text(0.4, 0.02, 'Predicted label', ha='left')
    f.colorbar(D.im_, ax=axes)
    #plt.show()
    return

def SearchImage(model, X, Y, N, prediction):
    y_pred = convert(model, X)
    images=[]
    Y=Y.tolist()
    for digit in digits:
        i=0
        while Y.count(digit)>0 and i<N:
            ind = Y.index(digit)
            if (y_pred[ind] == Y[ind]) and prediction or (y_pred[ind] != Y[ind]) and not(prediction):
                i+=1
                images.append(np.asarray(X[ind].reshape(28, 28)).squeeze())
            Y[ind]=-1
    return images

def PlotExamples(model, X, Y, N, prediction):
    images = SearchImage(model, X, Y, N, prediction)
    f = plt.figure(figsize=(N, 10))
    if prediction == True:
        f.suptitle('True Prediction', fontsize=16)
    else:
        f.suptitle('False Prediction', fontsize=16)

    i=1
    for image in images:
        ax = f.add_subplot(10, N, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(image)
        i+=1

def ShowExamples(model, X, Y, N):
    PlotExamples(model, X, Y, N, True)
    PlotExamples(model, X, Y, N, False)
    plt.show()
