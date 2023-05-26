from KNN import *
from PCA import *
from dataset import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
I = 12, gives acc 86 with our function
I = 9, gives acc 86 with built-in function
"""
# Ours
# x_train, x_test, y_train, y_test = test_train_split(
#     test_size= 0.2, random_state= 12
# )

# Built-in
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size= 0.2, random_state= 9
)


pca = PCA(9)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

knn = KNN(5)
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)
print(accuracy_score(y_test, y_pred))

# print(f"Accuracy at i = {i} is: {accuracy_score(y_test, y_pred)}")

# knn = KNN(5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print(accuracy_score(y_test, y_pred))