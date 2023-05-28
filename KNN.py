import math

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Manhattan distance
    def manhattan_distance(self, point1, point2):
        return abs(point1 - point2)

    def predict(self, X):
        y_pred = []
        # Loop over all the list entries
        for i in range(len(X)):
            distances = []
            
            # Calculate the distance between every feature of the current
            # entry and the saved train entries
            for j in range(len(self.X_train)):
                current_distances = [self.manhattan_distance(X[i][k], self.X_train[j][k]) for k in range(len(X[i]))]
                
                # Sum all the distances and add it to the list with the class of
                # the current train entry.
                distance = sum(current_distances)
                distances.append((distance, self.y_train[j]))
            
            # Sort the list of distances to get the nearest k neighbors
            distances.sort()
            k_nearest_neighbors = distances[:self.k]

            # Calculate the prediction for the current entry using the most
            # repeated labels in the list.
            k_nearest_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
            y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))

        return y_pred

