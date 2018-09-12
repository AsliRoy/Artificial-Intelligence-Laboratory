import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def init_param(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    
    param = {"W1" : W1, "b1": b1,
                  "W2" : W2, "b2": b2}
    return param

def f_prop(X, Y, param):
    m = X.shape[1]
    W1 = param["W1"]
    W2 = param["W2"]
    b1 = param["b1"]
    b2 = param["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m
    return cost, cache, A2

def backward_prop(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 * (1- A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients
    
def update_param(param, grads, eta):
    param["W1"] = param["W1"] - eta * grads["dW1"]
    param["W2"] = param["W2"] - eta * grads["dW2"]
    param["b1"] = param["b1"] - eta * grads["db1"]
    param["b2"] = param["b2"] - eta * grads["db2"]
    return param
    
df=pd.read_csv("iris.csv",sep=",",names=["pl","pw","sl","sw","class"])
df['class'],class_names = pd.factorize(df['class'])

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]  
print(Y.unique())


#Test-train split
msk = np.random.rand(len(df)) < 0.7
X_train = X[msk]
x_test = X[~msk]
Y_train=Y[msk]
y_test=Y[~msk]

x_test=np.array(x_test)
y_test=np.array(y_test)


X_train=np.array(X_train)
Y_train=np.array(np.array([Y_train]).T)

#Print the test-train dataset
print X_train
print x_test
print Y_train
print y_test

n_h=4
n_x=4
n_y=3
param = init_param(n_x, n_h, n_y)
num_iterations = 100000
eta = 0.01
losses = np.zeros((num_iterations, 1))

for i in range(num_iterations):
    losses[i, 0], cache, A2 = f_prop(X, Y, param)
    grads = backward_prop(X, Y, cache)
    param = update_param(param, grads, eta)
                                   
cost, _, A2 = f_prop(X, Y, param)
pred = (A2 > 0.5) * 1.0
print(A2)
print(pred)
plt.figure()
plt.plot(losses)
plt.show()

