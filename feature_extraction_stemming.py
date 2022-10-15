import os
import collections
import numpy as np
import matplotlib.pyplot as plt
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

def remove_punctuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text
    
def read_document_words(filename,voc):
    #this function gives us the feature (bow) for the filename
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = remove_punctuation(text.lower())
    bow = np.zeros(len(voc))
    i=1
    ps = PorterStemmer()
    for w in text.split():
        
        words.append(ps.stem(w)) 
            index = voc[w]
            bow[index] += 1

    return words
    

classes = {'cardiovascular_pulmonary':0 , 'consult':1, 'gastroenterology':2,
           'general_medicine':3, 'neurology':4, 'orthopedic':5,
           'radiology':6, 'surgery':7}
voc = load_vocabulary('vocabulary_stem.txt')
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
np.savetxt('train_data.txt', data.astype(int), fmt='%i')

#---- loading test dataset

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
np.savetxt('test_data.txt', test_data.astype(int) , fmt='%i')
