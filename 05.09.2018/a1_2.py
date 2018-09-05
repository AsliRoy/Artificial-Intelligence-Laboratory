import pandas as pd
import numpy as np
import random
random.seed(123)
import math





df=pd.read_csv("iris.csv",sep=",",names=["pl","pw","sl","sw","classy"])

classy = {'Iris-setosa': 1,'Iris-versicolor':2,'Iris-virginica':3}

df.classy = [classy[item] for item in df.classy]

print df

train = df[:105]
test = df[105:150]

print train.shape
print test.shape

train_X=train.values[:,0:4]
train_y=train.values[:,-1]

test_X=test.values[:,0:4]
test_y=test.values[:,-1]

print train_X.shape
print train_y.shape
print test_X.shape
print test_y.shape


def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in xrange(len(B[0]))] for i in xrange(len(A))]    
    for i in xrange(len(A)):
        for j in xrange(len(B[0])):
            for k in xrange(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in xrange(len(B[0]))]
    for j in xrange(len(B[0])):
        for k in xrange(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in xrange(len(A))]
    for i in xrange(len(A)):
        for j in xrange(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in xrange(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in xrange(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 4, 3] # number of neuron each layer

# Initiate weight and bias with 0 value
weight = [[0 for j in xrange(neuron[1])] for i in xrange(neuron[0])]
weight_2 = [[0 for j in xrange(neuron[2])] for i in xrange(neuron[1])]
bias = [0 for i in xrange(neuron[1])]
bias_2 = [0 for i in xrange(neuron[2])]

print weight 
print weight_2
print bias 
print bias_2

# Initiate weight with random between -1.0 ... 1.0
for i in xrange(neuron[0]):
    for j in xrange(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in xrange(neuron[1]):
    for j in xrange(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1


print weight 
print weight_2
for e in xrange(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        print train_y[idx]
        target[2] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in xrange(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2 
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in xrange(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in xrange(neuron[1]):
            for j in xrange(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in xrange(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in xrange(neuron[0]):
            for j in xrange(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
    
    cost_total /= len(train_X)
    if(e % 100 == 0):
        print cost_total

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x:x[1])[0])

# Print prediction
print preds

# Calculate accuration
acc = 0.0
for i in xrange(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print acc / len(preds) * 100, "%"