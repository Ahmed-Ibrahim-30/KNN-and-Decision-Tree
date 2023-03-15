# %%
import pandas
import numpy as np
import math
from sklearn.model_selection import train_test_split

# %%
dataset = pandas.read_csv("BankNote_Authentication.csv")

# shuffle All Rows & split the dataset
x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=['class']), dataset['class'], test_size=0.3)
# %%
x_train = np.array(x_train).reshape(-1, 4)
x_test = np.array(x_test).reshape(-1, 4)
y_train = np.array(y_train)
y_test = np.array(y_test)
# %%
# mean & standard Deviation for X Train
mean = np.mean(x_train)
std = np.std(x_train)


# %%
# alldata = (xtrain | xtest)
def normalize_feature(feature_num, alldata):
    arr = alldata[:, [feature_num]]
    update_data = np.zeros(arr.shape)
    for i in range(len(arr)):
        update_data[i] = (arr[i] - mean) / std
    return update_data


def measureAccuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


# %%
# normalize each feature
for i in range(4):
    x_train[:, [i]] = normalize_feature(i, x_train)
    x_test[:, [i]] = normalize_feature(i, x_test)


# %%
def Knn(k):
    y_pred = np.zeros(y_test.shape)
    for i in range(len(x_test)):
        dist_class = np.zeros(3)
        dist_class = np.resize(dist_class, (len(x_train), 3))  #
        for g in range(len(x_train)):
            one = pow((x_test[i][0] - x_train[g][0]), 2)
            two = pow((x_test[i][1] - x_train[g][1]), 2)
            three = pow((x_test[i][2] - x_train[g][2]), 2)
            four = pow((x_test[i][3] - x_train[g][3]), 2)
            distance = math.sqrt(one + two + three + four)
            dist_class[g][0] = distance
            dist_class[g][1] = y_train[g]
            dist_class[g][2] = g
        dist_class = dist_class[dist_class[:, 0].argsort()]  # sort all distance to choose best k elements
        zero = 0
        one = 0
        sum_zero_index = 0
        sum_one_index = 0
        enter1 = False
        enter2 = False
        for w in range(k):
            if dist_class[w][1] == 0:
                zero = zero + 1
                if not enter1:
                    enter1 = True
                    sum_zero_index += dist_class[w][2]
            if dist_class[w][1] == 1:
                one += 1
                if not enter2:
                    enter2 = True
                    sum_one_index += dist_class[w][2]

        if zero > one:
            y_pred[i] = 0
        elif zero < one:
            y_pred[i] = 1
        else:
            if sum_zero_index > sum_one_index:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred


# %%
for i in range(10):
    y_pred = Knn(i + 1)
    print("k = ", (i + 1))
    print("Number of correctly classified instances : ", np.sum(y_pred == y_test), "Total number of instances : ",
          len(y_test))
    print("Accuracy : ", measureAccuracy(y_pred, y_test))
    print("")
# %%

# %%

# %%
