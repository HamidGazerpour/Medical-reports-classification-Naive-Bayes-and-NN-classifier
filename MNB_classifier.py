import numpy as np
import build_vocabulary
from build_vocabulary import export_voc
import build_vocabulary_stemming
from build_vocabulary_stemming import export_voc_stem
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

classes = {'cardiovascular_pulmonary':0 , 'consult':1, 'gastroenterology':2,
           'general_medicine':3, 'neurology':4, 'orthopedic':5,
           'radiology':6, 'surgery':7}

train_data = np.loadtxt("train_stop_1000.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
w, b = train_mnb(Xtrain, Ytrain)

test_data = np.loadtxt("test_stop_1000.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

#----Accuracy testing
train_predictions = inference_mnb(Xtrain, w, b)
train_accuracy = (train_predictions == Ytrain).mean() * 100
# print ("Training accuracy = ", accuracy)

test_predictions = inference_mnb(Xtest, w, b)
test_accuracy = (test_predictions == Ytest).mean() * 100

print('accuracy on train dataset:',train_accuracy
      , 'accuracy on test dataset:',test_accuracy)
#-----------------------------------------------

# confusion matrix
cm = confusion_matrix(Ytest, test_predictions)

cm_display = ConfusionMatrixDisplay(cm).plot()
