from modules.preprocessing import clean_text, LabelEncoding, build_vocabulary
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
from models.decision_tree import DecisionTree
from modules.vectorizer import Vectorizer
import warnings
from stopwordsiso import stopwords as stopwords
import time
import sys

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
#vocabulary = build_vocabulary()

vectorizer = TfidfVectorizer(sublinear_tf = True, norm = 'l2', ngram_range = (1,2), stop_words = stopwords('ny'))
train_features = vectorizer.fit_transform(train_texts).toarray()
#test_features = vectorizer.transform(test_texts).toarray()
reduced_vocabulary = []
print('Transformed features shape: ',train_features.shape)
label_ids = train_data['Label_Id']

K = 600
for label_id, label in sorted(encoded_labels.items()):
    train_features_chi2 = chi2(train_features, label_ids == label_id)
    indices = np.argsort(train_features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]

    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    #print("# '{}':".format(label))
    #print("\t. Most correlated unigrams:\n\t\t. {}".format('\n\t\t. '.join(unigrams[-K:])))
    #print("\t. Most correlated bigrams:\n\t\t. {}".format('\n\t\t. '.join(bigrams[-K:])))
    reduced_vocabulary = reduced_vocabulary + unigrams[-K:] + bigrams[-K:]

reduced_vocabulary = list(set(reduced_vocabulary))
vectorizer = TfidfVectorizer(sublinear_tf = True, norm = 'l2', ngram_range = (1,2), stop_words = stopwords('ny'),
        vocabulary = reduced_vocabulary)

train_features_best = vectorizer.fit_transform(train_texts).toarray()
test_features_best = vectorizer.transform(test_texts).toarray()
print('\nReduced vocabulary features: ', train_features_best.shape, test_features_best.shape)

if len(sys.argv)==1:
    print('No Model Selected :-(\n')
else:
    if sys.argv[1] == 'dtree':
        dtree = DecisionTree(train_features_best, train_data['Label_Id'], test_data['ID'], le)
        dtree.predict_and_save_csv(test_features_best)
    elif sys.argv[1] == 'xgb':
        xgb = XGBoost(train_features_best, train_data['Label_Id'], test_data['ID'], le)
        xgb.predict_and_save_csv(test_features_best)
    elif sys.argv[1] == 'rdf':
        rdf = RandomForest(train_features_best, train_data['Label_Id'], test_data['ID'], le)
        rdf.predict_and_save_csv(test_features_best)

et = time.time()
print('\nMinutes elapsed:',(et - st) * 60 / 3600,'\n')
