import collections
import os
import nltk
from nltk.stem import PorterStemmer

def remove_punctuation(text):
    punc = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789"
    for p in punc:
        text = text.replace(p , " ")
    return text


def read_document(filename, s_w, stop_words = False ):
    f = open(filename, encoding="utf8")
    text = f.readlines()[0].rstrip()
    f.close()
    words = []
    text = remove_punctuation(text.lower())
    
    for w in text.split():
        if len(w) > 2:
            if stop_words:
                if w not in s_w:
                    words.append(w)
            else:
                words.append(w)
    return words

def read_document_stem(filename, s_w, stop_words = False ):
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    words = []
    text = remove_punctuation(text.lower())
    
    for w in text.split():
        if len(w) > 2:
            if stop_words:
                if w not in s_w:
                    words.append(w)
            else:
                words.append(w)
    ps = PorterStemmer()

    stem = [ps.stem(word) for word in words]
    return stem

def write_vocabulary(voc, filename,n):
    f = open(filename, "w")
    for word, count in sorted(voc.most_common(n)):
        print(word , file=f)
    f.close()
    
        
def export_voc(vocsize, stop_words=True): 
    if stop_words:
        s = open('stopwords.txt', encoding="utf8")
        s_w = s.read()
        s.close()
        s_w = remove_punctuation(s_w.lower())
    else:
        s_w = None
    voc = collections.Counter()
    for f in os.listdir("medical-reports/train"):
        voc.update(read_document("medical-reports/train/" + f, s_w, stop_words))
    
    write_vocabulary(voc, "vocabulary_summary_1000.txt", vocsize)
    return None

def export_voc_stem(vocsize, stop_words=True): 
    if stop_words:
        s = open('stopwords.txt', encoding="utf8")
        s_w = s.read()
        s.close()
        s_w = remove_punctuation(s_w.lower())
    else:
        s_w = None
    voc = collections.Counter()
    for f in os.listdir("medical-reports/train"):
        voc.update(read_document_stem("medical-reports/train/" + f, s_w, stop_words))
    
    write_vocabulary(voc, "vocabulary_summary_1000.txt", vocsize)
    return None
