import math

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # TODO: Implement me!!!!!!
    def calc_distance():
        pass

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            distances = []
            for j in range(len(self.X_train)):
                # Euclidean
                # distance = math.sqrt(sum([(X[i][k] - self.X_train[j][k])**2 for k in range(len(X[i]))]))
                # Manhattan
                # TODO: Use the implemented method
                distance = sum([abs((X[i][k] - self.X_train[j][k])) for k in range(len(X[i]))])
                distances.append((distance, self.y_train[j]))
            distances.sort()
            k_nearest_neighbors = distances[:self.k]
            k_nearest_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
            y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
        return y_pred