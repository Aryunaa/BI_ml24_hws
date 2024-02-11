import numpy as np
from collections import Counter
import random
def most_frequent_value(binary_vector):
    # Подсчет частоты каждого значения в бинарном векторе
    counts = np.bincount(binary_vector)
    # Нахождение индекса наиболее частого значения
    most_frequent_index = np.argmax(counts)
    return most_frequent_index

# Пример использования
binary_vector = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0,0,0,0])
most_frequent_val = most_frequent_value(binary_vector)
#print("Самое частое значение в бинарном векторе:", most_frequent_val)

k_vector = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0,0,0,0,2,2,2,2,2,2, 2])
counts = Counter(k_vector)
            # Нахождение наиболее частого значения
#most_frequent_value = max(counts, key=counts.get)
print(counts)
most_common_values = counts.most_common()
print(most_common_values)
most_frequent_value, highest_frequency = most_common_values[0]
if len(most_common_values) > 1 and most_common_values[1][1] == highest_frequency:
        # Собираем все самые частые значения с одинаковой частотой
        most_frequent_values = [value for value, count in most_common_values if count == highest_frequency]
        # Возвращаем случайное значение из этого набора
        most_frequent_value = random.choice(most_frequent_values)
print(most_frequent_value)


