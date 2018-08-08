import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from time import sleep

def plot_galary(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
def titles(y_predict, y_test, target_names):
    for i in range (y_predict.shape[0]):
        pred_name = target_names[y_predict[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

#load dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)
_, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
Y = lfw_dataset.target
target_names = lfw_dataset.target_names
#split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#compute a PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
#apply pca transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#dimensionallity has been reduced, we are ready to train our network
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, Y_train)

y_predict = clf.predict(X_test_pca)

print(classification_report(Y_test, y_predict, target_names=target_names))

prediction_titles = list(titles(y_predict, Y_test, target_names))
plot_galary(X_test, prediction_titles, h, w)