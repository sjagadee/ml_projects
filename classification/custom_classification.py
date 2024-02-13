import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

plt.style.use('fivethirtyeight')
import warnings


dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

new_feature = [5,7]

def k_nearest_neighbors(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    # knn algo
    distance = []
    for group in data:
        for feature in data[group]:
            # euclidean_distance = sqrt((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2)
            # euclidean_distance = np.sqrt(np.sum( (np.array(feature)-np.array(predict))**2 ))
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distance.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distance)[:k]]
    print(votes)
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbors(dataset, new_feature,k=3)
print(result)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color=result)
plt.show()