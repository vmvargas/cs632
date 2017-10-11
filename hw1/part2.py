#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import glob
import errno
import re
import io
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#loading raw emails data
raw_emails = []
path = './spam_data/0*.txt'
files = glob.glob(path)

for name in files:
    try: 
        with io.open(name, 'rb') as email:
            #remove XML and HTML code
            raw_email = BeautifulSoup(email,"lxml").text
            raw_email2 = BeautifulSoup(raw_email,"html.parser").text
            #remove numbers and special characters
            raw_email2 = re.sub('[^a-zA-Z ]+', ' ', raw_email2)            
            #remove stop words       
            raw_email3 = [w for w in word_tokenize(raw_email2) if not w in stopwords.words('english')]
            #remove word with len=1
            raw_email4 = [ w for w in raw_email3 if len(w) > 1 ]
            # Textblob detect part-of-speech (POS) tags and normalize words into their base form (lemmas)
            words = TextBlob(''.join(raw_email4)).words
            raw_email5 = [word.lemma for word in words]
            #joining the array of words to a string
            raw_email6 = ' '.join(map(str, raw_email5))            
            raw_email6 = unicode(raw_email6, 'utf8').lower()
            raw_emails.append([raw_email6])
            
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
          
raw_labels = []
with open('./spam_data/labels.txt','r') as labels:
    for line in labels:
        for label in line.split():
            if re.match('^\d$', label):
                raw_labels.append([label])

raw_content = pd.DataFrame(np.column_stack([raw_labels, raw_emails]),
                               columns=['label', 'content'])              

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


#split train and test data
#np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(raw_content['content'], raw_content['label'], test_size=0.5)

#start processing the train data 
# convert each message, represented as a list of tokens into a vector
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# TfidfTransformer does the counting, the term weighting and the normalization  
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#visualize the bow, basically a huge sparse matrix
#print ('sparse matrix shape:', X_train_tfidf.shape)
#print ('number of non-zeros:', X_train_tfidf.nnz)
#print ('sparsity: %.2f%%' % (100.0 * X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])))

#training KNN model
knn = KNeighborsClassifier()
knn.fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

y_pred = knn.predict(X_test_tfidf)
print ('Overall accuracy is: ' + '{:2f}'.format(accuracy_score(y_test, y_pred)))

print 'confusion matrix\n', confusion_matrix(y_test, y_pred)
print '(row=expected, col=predicted)'

plt.matshow(confusion_matrix(y_test, y_pred), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')