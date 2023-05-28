from KNN import *
from PCA import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
# With the sorted dataset:-
I = 9, gives acc 90 with built-in function

# With the shuffled dataset:-
I = 46, gives acc 88 with built-in function
"""
# Sorted dataset
# dataset = pd.read_csv("./hypertension_dataset.csv")

# Shuffled dataset
dataset = pd.read_csv("./shuffled_dataset.csv")

features = np.array(dataset.drop("target", axis=1))
classes = np.array(dataset["target"])

# accs = []
# for i in range(101):
    # x_train, x_test, y_train, y_test = train_test_split(
    #     features, classes, test_size= 0.2, random_state= i
    # )
x_train, x_test, y_train, y_test = train_test_split(
    features, classes, test_size= 0.2, random_state= 46
)


pca = PCA(9)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNN(5)
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)
print(accuracy_score(y_test, y_pred))
    # accs.append(accuracy_score(y_test, y_pred))

# print(f"Max = {max(accs)}, I = {accs.index(max(accs))}")


# knn = KNN(5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print(accuracy_score(y_test, y_pred))