from random import choice
from numpy import array, dot, random
training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]


print training_data



w=random.rand(3)
print w

x, expected = choice(training_data)
print x, expected
result = dot(w, x)
print result

x, expected = choice(training_data)
print x, expected
result = dot(w, x)
print result
