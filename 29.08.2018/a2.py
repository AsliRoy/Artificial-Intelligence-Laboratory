from random import choice
from numpy import array, dot, random
import numpy as np
import matplotlib.pyplot as plt

step_value = lambda x: 0 if x < 0 else 1

data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),
]

w = random.rand(3)
reference_array=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])
errors = []
alpha = 0.5 
epochs = 1000 
for i in xrange(epochs):
    x, expected = choice(data)
    result = dot(w, x)
    error = expected - step_value(result)
    errors.append(error)
    w = w+ alpha * error * x

result_list=[]
for x, _ in data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, step_value(result)))
    result_list.append(step_value(result))


for d, sample in enumerate(reference_array):
   
    
    if result_list[d] < 1:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

plt.plot([0,1.2],[1.2,0])
plt.show()
plt.close()
