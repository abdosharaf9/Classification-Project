import csv
import random
import numpy as np


# TODO: change it!!!!!!!
_dataset = open("./hypertension_data - Copy.csv", "r")
dataset = list(csv.reader(_dataset, delimiter=","))

data = []
classes = []

for row in dataset:
    row_data = [float(feature) for feature in row]
    data.append(row_data[:-1])
    classes.append(row_data[-1])

X = np.array(data)
Y = np.array(classes)


# TODO: Change it!!!!!!
def test_train_split(test_size, random_state):
    random.seed(random_state)
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    length=len(Y)
    train_size=length-(length*test_size)
    r=random.sample(range(0,length),length)
    for i in range(length):
        if i<train_size:
            x_train.append(X[r[i]])
            y_train.append(Y[r[i]])
        else:
            x_test.append(X[r[i]])
            y_test.append(Y[r[i]])

    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

