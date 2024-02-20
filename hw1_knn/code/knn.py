import numpy as np
from collections import Counter
import random

class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        test = X
        num_test = test.shape[0]
        num_train = self.train_X.shape[0]
        distances = np.zeros((num_test, num_train))
        
        # Для каждого изображения из теста
        for i in range(num_test):
            # для каждого изображения из трейна
            for j in range(num_train):
                # Вычисляем расстояние Манхэттена между i-м изображении из test и j-м изображением из train
                distances[i, j] = np.sum(np.abs(test[i] - self.train_X[j]))

        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        test = X
        num_test = test.shape[0]
        num_train = self.train_X.shape[0]
        distances = np.zeros((num_test, num_train))
        # Для каждого изображения из теста
        for i in range(num_test):
            # Для каждого изображения из теста вычисляем абсолютную разницу с каждым изображением из трейна,
            # а затем суммируем значения  для каждого элемента трейна
            #  (получаем вектор расстояний между i изображением и j изображением , вектор с размером num_train)


            #distances[i] содержит расстояния Манхэттена между test[i] и всеми векторами из тренировочной выборки
            distances[i] = np.sum(np.abs(test[i] - self.train_X), axis=1)

        return distances       


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        
        test = X
        
        #distances = np.sum(np.abs(test - self.train_X))
        distances = np.sum(np.abs(test[:, np.newaxis, :] - self.train_X), axis=2)
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        mins_dists_indexes = min_indices_along_rows(distances, self.k) # for each row (test)
        for i in range(n_test): # по всем строкам (test)
            min_dist_indexes = mins_dists_indexes[i]
            k_vector = self.train_y[min_dist_indexes] # вектор из трейнов для теста
            prediction[i] = most_frequent_value(k_vector)

        return prediction

    
    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        
        n_test = distances.shape[0]
        prediction = np.empty(n_test)
        mins_dists_indexes = min_indices_along_rows(distances, self.k) # for each row (test)
        for i in range(n_test): # по всем строкам (test)
            min_dist_indexes = mins_dists_indexes[i]
            k_vector = self.train_y[min_dist_indexes] # вектор из трейнов для теста

            # Подсчет частоты каждого значения в векторе
            counts = Counter(k_vector)
            # Нахождение наиболее частого значения
            #print(counts)
            most_common_values = counts.most_common()
            #print(most_common_values)
            most_frequent_value, highest_frequency = most_common_values[0]
            
            if len(counts) > 1 and counts.most_common()[0][1] == counts.most_common()[1][1]:
                # Собираем все самые частые значения с одинаковой частотой
                most_frequent_values = [value for value, count in counts.items() if count == highest_frequency]
                # Возвращаем случайное значение из этого набора
                #print(most_common_values)
                most_frequent_value = random.choice(most_frequent_values)    
                #print(most_frequent_value)
            #print(type(most_frequent_value))
            #print(type(prediction[i]))
            prediction[i] = most_frequent_value
            
            #print('prediction = ' + str(prediction[i]))
            #print(prediction[i])

        prediction = prediction.astype('int').astype('str')  
        #for i in range(n_test):
             #print('prediction = ' + prediction[i])
             #print(prediction[i])
        return prediction


def min_indices_along_rows(matrix, n):
    # Находим порядок элементов, отсортированных по каждой строке
    sorted_indices = np.argsort(matrix, axis=1)
    # Берем первые n индексов из каждой строки (т.е. индексы минимальных значений)
    #min_indices = sorted_indices[:, :n]
    return sorted_indices[:,:n]


def most_frequent_value(binary_vector):
    # Подсчет частоты каждого значения в бинарном векторе
    counts = np.bincount(binary_vector)
    # Нахождение индекса наиболее частого значения
    most_frequent_index = np.argmax(counts)
    return most_frequent_index
