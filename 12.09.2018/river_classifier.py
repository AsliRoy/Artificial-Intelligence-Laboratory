import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def neuralnet_train(x_train, y_train):
    iterations = 1000;
    activation = 'logistic';
    mlp = MLPClassifier(solver='lbfgs', alpha=0.001, activation=activation, hidden_layer_sizes=(5, 5), max_iter=iterations);
    mlp.fit(x_train, y_train);
    return mlp;

def neuralnet_test(x_test, y_test, y_pred, mlp):
    y_pred = mlp.predict(x_test);
    acc = accuracy_score(y_test, y_pred, normalize=False)/len(y_test);
    print("Test accuracy: ", acc);
    print(y_pred);
    return y_pred;

def open_image(file, x):
    img = Image.open(file).convert('L');
    x = np.array(img);
    x = np.reshape(x, (len(x)*len(x[0]), 1));
    return x;
    
def proc_image(x, y):
    for i in range(len(x)):
        if x[i]>245:
            y.append(1);
        else:
            y.append(0);            
    return y;
            

    
def create_image_file(y_pred):
    y_img = [];
    for i in range(len(y_pred)):
        if y_pred[i]>=0.5:
            y_img.append(255);
        else:
            y_img.append(0);
    y_img = np.array(y_img);
    
    print("Ouput image file saved successfully");
    img_size = (64, 64);
    new_image = Image.new('L', img_size);
    new_image.putdata(y_img);
    new_image.save('river_pred.png');


dataset = [];
x_train = [];
x_test = [];
y_train = [];
y_test = [];
y_pred = [];
    
ip1 = raw_input("Enter input image file path: ");
ip2 = raw_input("Enter test image file path: ");
x_train = open_image(ip1, x_train);
y_train = proc_image(x_train, y_train);
mlp = neuralnet_train(x_train, y_train);
x_test = open_image(ip2, x_test);
y_test = proc_image(x_test, y_test);
y_pred = neuralnet_test(x_test, y_test, y_pred, mlp);
create_image_file(y_pred);