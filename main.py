from KNN import *
from PCA import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Get the features without the classification
features = np.array(dataset.drop("target", axis=1))

# Get the classification data
classes = np.array(dataset["target"])

# Split the dataset into train and test portions
x_train, x_test, y_train, y_test = train_test_split(
    features, classes, test_size= 0.2, random_state= 46
)


# Reduce the features to 9 using PCA
pca = PCA(9)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)


# Classify the data using KNN with k = 5
knn = KNN(5)
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)

# Print the accuracy
print(f"Accuracy at k = 5 is: {accuracy_score(y_test, y_pred) * 100}%\n")


# Get the accuracy for other k values from 1 to 10
pred_list = []
for k in range(1,11):
    knn = KNN(k)
    knn.fit(x_train_pca, y_train)
    y_pred = knn.predict(x_test_pca)

    pred_list.append(accuracy_score(y_pred, y_test) * 100)
    print(f"Accuracy at k = {k} is: {accuracy_score(y_test, y_pred) * 100}%")

# Line graph
plt.plot(range(1, 11), pred_list)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores for Different Values of K')
plt.show()


# Plotting the accuracy scores as bars & Change the color of the most
# valued one to red.
bars = plt.bar(range(1, 11), pred_list)
max_index = np.argmax(np.array(pred_list))
bars[max_index].set_color('red')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores for Different Values of K')

# Write the accuracy on the top of the bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom', size=8)


plt.show()

