import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

folder='./twitter-datasets-pp/'
with open(folder+'train_arrays.pkl', 'rb') as f:
    train_arrays = pickle.load(f)

with open(folder+'train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

with open(folder+'test_arrays.pkl', 'rb') as f:
    test_arrays = pickle.load(f)

classifier = svm.SVC()
classifier.fit(train_arrays, train_labels)
predictions = classifier.predict(test_arrays)
print(np.where(predictions==1))
print(predictions.shape)
folder_sub = './submissions/'
filename = folder_sub+'submission3.csv'
f = open(filename,'w')
header = 'Id,Prediction\n'
f.write(header)
for i in range(len(predictions)):
    id = str(i+1)
    f.write(id + ',' + str(int(predictions[i])) + '\n')

f.close()
