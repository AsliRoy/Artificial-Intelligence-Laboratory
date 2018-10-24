from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def main():
	data=np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
	X=data[:,0:2]
	Y=data[:,2]
	clf = svm.SVC(kernel='rbf',degree=2)
	clf.fit(X, Y)  

	print("Intercept:")
	print(clf.intercept_)
	print("Weights:")
	print(clf.dual_coef_[0])
	h=0.02
	fignum=2
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z=Z.reshape(xx.shape)
	plt.figure(fignum, figsize=(4, 3))
	#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
	                levels=[-.5, 0, .5])
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.axis('tight')
	plt.show()

if __name__ == "__main__":
	main()