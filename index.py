from modules.preprocessing import clean_text, LabelEncoding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from models.svc import SVC
from models.linear_svm import LinearSVM
from models.naive_bayesian import NaiveBayesian
from models.logistic import Logistic
from models.xgboost import XGBoost
from models.random_forest import RandomForest
from modules.vectorizer import Vectorizer
import warnings
from stopwordsiso import stopwords as stopwords
import time

st = time.time()
warnings.filterwarnings('ignore')
#Reading from dataset
train_path = 'input/Train.csv'
test_path = 'input/Test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
train_texts = train_data['Text']
test_texts = test_data['Text']
all_labels = train_data['Label'].unique()
label_count = []
for label in all_labels:
    mask = train_data['Label'] == label
    label_count.append((label,len(train_data[mask])))

le = LabelEncoding(all_labels)
train_data, encoded_labels = le.encode(train_data)
data_stats = pd.DataFrame(label_count, columns=['category','no of labels'])
data_stats.plot(x='category',y='no of labels', kind='bar', legend=False, grid=True, figsize=(8, 8))
plt.title('Number of comments per category')
plt.ylabel('No of occurences')
plt.xlabel('Category')
#plt.show()
print()

'''
vectorizer = TfidfVectorizer(sublinear_tf = True, norm = 'l2', ngram_range = (1,2), stop_words = stopwords('ny'))
train_features = vectorizer.fit_transform(train_texts).toarray()
test_features = vectorizer.transform(test_texts).toarray()
print('Transformed features shape: ',train_features.shape)
label_ids = train_data['Label_Id']

K = 750
kbest = SelectKBest(chi2, k = K)
train_features_best = kbest.fit_transform(train_features, train_data['Label_Id'])
test_features_best = kbest.transform(test_features)
print('\nReduced chi2 features: ', train_features_best.shape, test_features_best.shape)

#linear_model = LinearSVM(train_features_best, train_data['Label_Id'], test_data['ID'], le)
#linear_model.predict_and_save_csv(test_features_best)

'''

logistic_model = Logistic(train_texts, train_data['Label_Id'], test_data['ID'], le)
logistic_model.predict_and_save_csv(test_texts)

et = time.time()
print('\nMinutes elapsed:',(et - st) * 60 / 3600,'\n')
