## Training a network and Testing it on test dataset

import time
from os.path import exists

import numpy as np
import torch
import pandas as pd
import model
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import train_loader, device, model, error, optimizer, train_loader, test_loader,output_label
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import sklearn.metrics as metrics
from itertools import chain

################################################################################################################
def test_model(model, device):
    CM = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            valx, valy = data[0].to(device), data[1].to(device)
            val_pred = model(valx)
            preds = torch.argmax(val_pred.data, 1)
            CM += confusion_matrix(valy.cpu(), preds.cpu(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matrix : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)
        print('- Precision: ', (tp / (tp + fp)) * 100)
        print('- NPV: ', (tn / (tn + fn)) * 100)
        print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
        print()

    return acc, CM
############################################################################################################
num_epochs = 1
count = 0
# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

if not exists("fashion_28.pth"):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

            # Testing the model

            
            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

# %% md
    torch.save(model, 'fashion_28.pth')

### Visualizing the Loss and Accuracy with Iterations
else:
    
    print("Model available... loading...")
    # Loading the saved model
    y_pred_list = []
    model = torch.load("fashion_28.pth")
    model.to(device)


    with torch.no_grad():
        model.eval()
"""        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    confusion_matrix(y_test, y_pred_list)
    print(classification_report(y_test, y_pred_list))"""

""" for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    train = Variable(images.view(100, 1, 28, 28))
    labels = Variable(labels)
    #test_model(model, device)
    outputs = model(train)
    loss = error(outputs, labels)

    # Initializing a gradient as 0 so there is no mixing of gradient among the batches
    optimizer.zero_grad()

    # Propagating the error backward
    loss.backward()

    # Optimizing the parameters
    optimizer.step()"""

        count += 1

    # Testing the model

                
        if not (count % 50):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)

            test = Variable(images.view(100, 1, 28, 28))

            outputs = model(test)

            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()

            total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))"""


# %%

plt.plot(iteration_list, loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()

# %%

plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()

# %% md

### Looking the Accuracy in each class of FashionMNIST dataset

# %%

class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test = Variable(images)
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()

        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1

for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

# %% md

### Printing the Confusion Matrix

# %%
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

# %%


confusion_matrix(labels_l, predictions_l)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))

