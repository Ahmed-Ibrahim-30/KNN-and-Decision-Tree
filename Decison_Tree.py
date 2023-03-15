#%%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot
from sklearn import tree
#%%
data = pd.read_csv("BankNote_Authentication.csv")
#data = (data - data.min()) / (data.max() - data.min())

#["variance" => "entropy"] for plotting
feature_names = data.columns.drop('class').to_numpy()
#["class"] for plotting
target_names = ['forged', 'real']
#%%
# Drop class (Y)
X = data.drop('class', axis=1).to_numpy()
#%%
# class
y = data["class"].to_numpy()
#%%
test_size = 0.75
Accuracy = []
TreeSize = []
for i in range(5):
    # Creating Train and Test datasets & split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    #Classifier Model
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)

    # Predict Accuracy Score
    y_pred = clf.predict(X_test)
    print("Testing: {}% , Training: {}%".format(round(test_size * 100, 1), round((1 - test_size) * 100, 1)))
    print("Train Data Accuracy: {} , Test Data Accuracy: {}%".format(
        accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)),
        round(accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3)))
    print("Size Of tree : {}".format(clf.tree_.node_count))
    print("Error: {}%".format(round(100 - accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3)))
    print("")

    TreeSize.append(clf.tree_.node_count)
    Accuracy.append(round(accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3))

    #fig = plot.figure(figsize=(40,20))
    #_ = tree.plot_tree(clf, feature_names=feature_names,class_names=target_names,filled=True)

print("==> Testing: {}% , Training: {}%".format(round(test_size * 100, 1), round((1 - test_size) * 100, 1)))
print(
    "==> Max Data Accuracy: {}% , Min Data Accuracy: {}% , Avg Data Accuracy: {}%".format(max(Accuracy), min(Accuracy),
                                                                                          np.average(Accuracy)))
print("==> Max Size Of tree: {} , Min Size Of tree: {} , Avg Size Of tree: {}".format(max(TreeSize), min(TreeSize),
                                                                                  round(np.average(TreeSize), 3)))
print("")
#%%
reportAvgAccuracy = []
reportTrainSize = []
reportAvgNodeSize = []
test_size = 0.70
for i in range(5):
    reportAvgMean = []
    Accuracy = []
    TreeSize = []
    for j in range(5):
        # Creating Train and Test datasets & split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        #Classifier Model
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X_train, y_train)

        # Predict Accuracy Score
        y_pred = clf.predict(X_test)
        print("Iteration : {}".format(j + 1))
        print("Testing: {}% , Training: {}%".format(round(test_size * 100, 1), round((1 - test_size) * 100, 1)))
        print('Mean: {}'.format(np.mean(y_pred)))
        print("Size Of tree : {}".format(clf.tree_.node_count))
        print("Train Data Accuracy: {} , Test Data Accuracy: {}%".format(
            accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)),
            round(accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3)))
        print("Error: {}%".format(round(100 - accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3)))
        print("")

        # Avg Operations
        TreeSize.append(clf.tree_.node_count)
        Accuracy.append(round(accuracy_score(y_true=y_test, y_pred=y_pred) * 100, 3))
        reportAvgMean.append(np.mean(y_pred))

        #fig = plot.figure(figsize=(40,20))
        #_ = tree.plot_tree(clf, feature_names=feature_names,class_names=target_names,filled=True)
    print("==> Testing: {}% , Training: {}%".format(round(test_size * 100, 1), round((1 - test_size) * 100, 1)))
    print("==> Max Data Accuracy: {}% , Min Data Accuracy: {}% , Avg Data Accuracy: {}%".format(max(Accuracy),
                                                                                                min(Accuracy),
                                                                                                np.average(Accuracy)))
    print("==> Max Size Of tree: {} , Min Size Of tree: {} , Avg Size Of tree: {}".format(max(TreeSize), min(TreeSize),
                                                                                          round(np.average(TreeSize),
                                                                                                3)))
    print("==> Average Mean: {}".format(round(np.average(reportAvgMean), 4)))

    reportAvgAccuracy.append(np.average(Accuracy))
    reportTrainSize.append(round((1 - test_size) * 100, 1))
    reportAvgNodeSize.append(np.average(TreeSize))
    test_size = test_size - 0.10
    print("")
#%%
plot.plot(reportTrainSize, reportAvgAccuracy, color='red')
plot.title('Avg Accuracy Over Each TrainSize')
plot.xlabel('TrainSize')
plot.ylabel('Accuracy')
plot.grid(True)
plot.show()
#%%
plot.plot(reportTrainSize, reportAvgNodeSize, color='green')
plot.title('Avg Size Of Tree Over Each TrainSize')
plot.xlabel('TrainSize')
plot.ylabel('Size Of Tree')
plot.grid(True)
plot.show()
#%%
