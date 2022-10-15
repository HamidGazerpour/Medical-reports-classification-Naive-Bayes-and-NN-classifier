import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import build_vocabulary
from build_summary_vocabulary import export_voc
from build_summary_vocabulary import export_voc_stem
import nltk
from nltk.stem import PorterStemmer

def load_vocabulary(file_name):
    f = open(file_name)
    voc ={}
    n = 0
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.closed
    return voc

def read_document_stem(filename, s_w, stop_words = False ):
    #this function gives us the feature (bow) for the filename
    f = open(filename, encoding="utf8")
    text = f.readlines()[0].rstrip()
    f.close()
    text = remove_punctuation(text.lower())
    bow = np.zeros(len(voc))
    ps = PorterStemmer()
    for w in text.split():
        if ps.stem(w) in voc:
            index = voc[ps.stem(w)]
            bow[index] += 1

    return bow

def remove_punctuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text
    
def read_document(filename,voc):
    #this function gives us the feature (bow) for the filename
    f = open(filename, encoding="utf8")
    text = f.readlines()[0].rstrip()
    f.close()
    text = remove_punctuation(text.lower())
    bow = np.zeros(len(voc))
    # pss = PorterStemmer()
    for w in text.split():
        # if pss.stem(w) in voc:
            # index = voc[pss.stem(w)]
            # bow[index] += 1
        if w in voc:
            index = voc[w]
            bow[index] += 1
    return bow

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

#---- main
classes = {'cardiovascular_pulmonary':0 , 'consult':1, 'gastroenterology':2,
           'general_medicine':3, 'neurology':4, 'orthopedic':5,
           'radiology':6, 'surgery':7}
#---- Stopwords off
export_voc(1000,False)
voc = load_vocabulary('vocabulary_summary_1000.txt')
n = len(voc)
m = len(os.listdir("medical-reports/train/"))
xtrain = np.zeros(m*n).reshape(m,n)

documents = []
labels = []

for f in os.listdir('medical-reports/train/'):
    documents.append(read_document('medical-reports/train/'+f,voc))
    labels.append(int(classes[f[5:-4]]))
    
X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt('train_summary.txt', data.astype(int), fmt='%i')

m = len(os.listdir("medical-reports/test/"))
xtrain = np.zeros(m*n).reshape(m,n)

test_docs = []
test_labels = []

for f in os.listdir('medical-reports/test/'):
    test_docs.append(read_document('medical-reports/test/'+f,voc))
    test_labels.append(int(classes[f[5:-4]]))
    
X = np.stack(test_docs)
Y = np.array(test_labels)
test_data = np.concatenate([X,Y[:,None]],1)
np.savetxt('test_summary.txt', test_data.astype(int) , fmt='%i')

train_data = np.loadtxt("train_summary.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
w, b = train_mnb(Xtrain, Ytrain)

test_data = np.loadtxt("test_summary.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

train_predictions = inference_mnb(Xtrain, w, b)
train_accuracy = (train_predictions == Ytrain).mean() * 100

test_predictions = inference_mnb(Xtest, w, b)
test_accuracy = (test_predictions == Ytest).mean() * 100
print('')
print('accuracy on train dataset:',train_accuracy
      , 'accuracy on test dataset:',test_accuracy)

#----  Stopwords on
export_voc(1000,True)
voc = load_vocabulary('vocabulary_summary_1000.txt')
n = len(voc)
m = len(os.listdir("medical-reports/train/"))
xtrain = np.zeros(m*n).reshape(m,n)

documents = []
labels = []

for f in os.listdir('medical-reports/train/'):
    documents.append(read_document('medical-reports/train/'+f,voc))
    labels.append(int(classes[f[5:-4]]))
    
X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt('train_summary.txt', data.astype(int), fmt='%i')

m = len(os.listdir("medical-reports/test/"))
xtrain = np.zeros(m*n).reshape(m,n)

test_docs = []
test_labels = []

for f in os.listdir('medical-reports/test/'):
    test_docs.append(read_document('medical-reports/test/'+f,voc))
    test_labels.append(int(classes[f[5:-4]]))
    
X = np.stack(test_docs)
Y = np.array(test_labels)
test_data = np.concatenate([X,Y[:,None]],1)
np.savetxt('test_summary.txt', test_data.astype(int) , fmt='%i')

train_data = np.loadtxt("train_summary.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
w, b = train_mnb(Xtrain, Ytrain)

test_data = np.loadtxt("test_summary.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

train_predictions = inference_mnb(Xtrain, w, b)
train_accuracy = (train_predictions == Ytrain).mean() * 100

test_predictions = inference_mnb(Xtest, w, b)
test_accuracy = (test_predictions == Ytest).mean() * 100
print('Considering Stopwords: \n')
print('accuracy on train dataset:',train_accuracy
      , 'accuracy on test dataset:',test_accuracy)

#---- Stopwords off -- stemming on
export_voc_stem(1000,False)
voc = load_vocabulary('vocabulary_summary_1000.txt')
n = len(voc)
m = len(os.listdir("medical-reports/train/"))
xtrain = np.zeros(m*n).reshape(m,n)
del documents,labels
documents = []
labels = []

for f in os.listdir('medical-reports/train/'):
    documents.append(read_document_stem('medical-reports/train/'+f,voc))
    labels.append(int(classes[f[5:-4]]))
    
X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt('train_summary.txt', data.astype(int), fmt='%i')

m = len(os.listdir("medical-reports/test/"))

test_docs = []
test_labels = []

for f in os.listdir('medical-reports/test/'):
    test_docs.append(read_document_stem('medical-reports/test/'+f,voc))
    test_labels.append(int(classes[f[5:-4]]))
    
X = np.stack(test_docs)
Y = np.array(test_labels)
test_data = np.concatenate([X,Y[:,None]],1)
np.savetxt('test_summary.txt', test_data.astype(int) , fmt='%i')

train_data = np.loadtxt("train_summary.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
w, b = train_mnb(Xtrain, Ytrain)

test_data = np.loadtxt("test_summary.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

train_predictions = inference_mnb(Xtrain, w, b)
train_accuracy = (train_predictions == Ytrain).mean() * 100

test_predictions = inference_mnb(Xtest, w, b)
test_accuracy = (test_predictions == Ytest).mean() * 100
print('considering stemming: \n')
print('accuracy on train dataset:',train_accuracy
      , 'accuracy on test dataset:',test_accuracy)

#----  Stopwords on ---- stemming on
export_voc_stem(1000,True)
voc = load_vocabulary('vocabulary_summary_1000.txt')
n = len(voc)
m = len(os.listdir("medical-reports/train/"))
xtrain = np.zeros(m*n).reshape(m,n)

documents = []
labels = []

for f in os.listdir('medical-reports/train/'):
    documents.append(read_document_stem('medical-reports/train/'+f,voc))
    labels.append(int(classes[f[5:-4]]))
    
X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt('train_summary.txt', data.astype(int), fmt='%i')

m = len(os.listdir("medical-reports/test/"))
xtrain = np.zeros(m*n).reshape(m,n)

test_docs = []
test_labels = []

for f in os.listdir('medical-reports/test/'):
    test_docs.append(read_document_stem('medical-reports/test/'+f,voc))
    test_labels.append(int(classes[f[5:-4]]))
    
X = np.stack(test_docs)
Y = np.array(test_labels)
test_data = np.concatenate([X,Y[:,None]],1)
np.savetxt('test_summary.txt', test_data.astype(int) , fmt='%i')

train_data = np.loadtxt("train_summary.txt" , dtype ='int')
Xtrain = train_data[:, :-1]
Ytrain = train_data[:, -1]
w, b = train_mnb(Xtrain, Ytrain)

test_data = np.loadtxt("test_summary.txt" , dtype ='int')
Xtest = test_data[:, :-1]
Ytest = test_data[:, -1]

train_predictions = inference_mnb(Xtrain, w, b)
train_accuracy = (train_predictions == Ytrain).mean() * 100

test_predictions = inference_mnb(Xtest, w, b)
test_accuracy = (test_predictions == Ytest).mean() * 100
print('Considering Stopwords and stemming: \n')
print('accuracy on train dataset:',train_accuracy
      , 'accuracy on test dataset:',test_accuracy)