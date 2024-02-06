import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

plt.style.use('fivethirtyeight')
import warnings


dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

new_feature = [5,7]

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color='g')
plt.show()