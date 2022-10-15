import numpy as np
import build_vocabulary
from build_vocabulary import export_voc
import build_vocabulary_stemming
from build_vocabulary_stemming import export_voc_stem
import matplotlib.pyplot as plt

def train_mnb(X, Y):
    m , n = X.shape
    k = Y.max() + 1
    
    probs = np.empty(( k , n ) )
    for c in range ( k ) :
        counts = X[Y == c,:]. sum (0)
        tot = counts.sum()
        probs[c , :] = ( counts + 1) / ( tot + n )
    priors = np.bincount(Y) / m
    W = np.log (probs) . T
    b = np.log (priors)
    return W , b
    

def inference_mnb(X, W, b):
    scores = X @ W + b.T
    labels = np. argmax (scores ,1)
    return labels


voc_size =[]
train_accuracy = []
test_accuracy = []
stop_words = True

for i in range(4):
    voc_size.append((i+1)*500)
    export_voc_stem(voc_size[i], stop_words)
    
    exec(open('feature_extraction.py').read())
    #----Loading train and test datasets
    train_data = np.loadtxt("train_data.txt" , dtype ='int')
    Xtrain = train_data[:, :-1]
    Ytrain = train_data[:, -1]
    w, b = train_mnb(Xtrain, Ytrain)
    
    test_data = np.loadtxt("test_data.txt" , dtype ='int')
    Xtest = test_data[:, :-1]
    Ytest = test_data[:, -1]
    
    #----Accuracy testing
    train_predictions = inference_mnb(Xtrain, w, b)
    train_accuracy.append( (train_predictions == Ytrain).mean() * 100)
    # print ("Training accuracy = ", accuracy)
    
    test_predictions = inference_mnb(Xtest, w, b)
    test_accuracy.append( (test_predictions == Ytest).mean() * 100)
    # print ("Test accuracy = ", accuracy)
plt.plot(voc_size, train_accuracy, label='Train accuracy')
plt.plot(voc_size, test_accuracy, label='Test accuracy')
plt.legend(loc='center right')
plt.xlabel('Vocabulary size')
plt.ylabel('Accuracy (%)')
print('highest accuracy on test dataset:', np.max(test_accuracy)
      , ', Best voc size', (np.argmax(test_accuracy)+1)*500)
#-----------------------------------------------

