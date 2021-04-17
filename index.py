from modules.preprocessing import clean_text, LabelEncoding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from modules.vectorizer import Vectorizer
import warnings

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
train_data = le.encode(train_data)
'''
data_stats = pd.DataFrame(label_count, columns=['category','no of labels'])
data_stats.plot(x='category',y='no of labels', kind='bar', legend=False, grid=True, figsize=(8, 8))
plt.title('Number of comments per category')
plt.ylabel('No of occurences')
plt.xlabel('Category')
#plt.show()
print()
'''
linear_model = LinearSVM(train_texts, train_data['Label_Id'], test_data['ID'], le)
linear_model.predict_and_save_csv(test_texts)