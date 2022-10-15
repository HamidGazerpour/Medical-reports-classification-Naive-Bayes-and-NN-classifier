import numpy as np
import pvml
import matplotlib.pyplot as plt
import random
import normalization as norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

train_data = np.loadtxt("train_summary.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]

test_data = np.loadtxt("test_summary.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]


# here we perform some feature normalization
#Xtrain, Xtest = norm.minmax_normalization(Xtrain, Xtest)
Xtrain = norm.l2_normalization(Xtrain)
Xtest = norm.l2_normalization(Xtest)

# Reduce the data to class 0-n
# n = 35
# Xtrain 	= Xtrain[Ytrain < n, :]
# Ytrain 	= Ytrain[Ytrain < n]
# Xtest 	= Xtest[Ytest < n, :]
# Ytest 	= Ytest[Ytest < n]


def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    # print(labels)
    return acc


# parameters of stochastic gradient descent (size of minibatches)
m , n = Xtrain.shape
b = 8
# plotting real progression of training and test accuracy
# plt.ion()
train_accs = []
test_accs = []
epochs = []

np.random.seed(0)
net = pvml.MLP([n, 8])
for epoch in range(400):
	net.train(Xtrain, Ytrain, 1e-4, steps=m // b, batch=b)
	if epoch % 5 == 0:
		train_acc = accuracy(net, Xtrain, Ytrain)
		test_acc = accuracy(net, Xtest, Ytest)
		print (epoch, train_acc * 100, test_acc * 100)
		train_accs.append(train_acc * 100)
		test_accs.append(test_acc * 100)
		epochs.append(epoch)
		plt.clf()
		plt.plot(epochs, train_accs)
		plt.plot(epochs, test_accs)
		plt.ylabel("Accuracy (%)")
		plt.xlabel("Epochs")
		plt.legend(["train", "test"])
		plt.pause(0.01)

plt.ioff()
plt.show()
print("Train acc: ", accuracy(net, Xtrain, Ytrain), "Test acc: ", accuracy(net, Xtest, Ytest))
# net.save("speech_mlp.npz") #saving parameters
labels, probs = net.inference(Xtest)
cm = confusion_matrix(Ytest, labels)

cm_display = ConfusionMatrixDisplay(cm).plot()

