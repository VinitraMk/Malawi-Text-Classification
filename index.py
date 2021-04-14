from modules.preprocessing import clean_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from models.svc import SVC
from models.linear_svm import LinearSVM
from modules.vectorizer import Vectorizer

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

data_stats = pd.DataFrame(label_count, columns=['category','no of labels'])
data_stats.plot(x='category',y='no of labels', kind='bar', legend=False, grid=True, figsize=(8, 8))
plt.title('Number of comments per category')
plt.ylabel('No of occurences')
plt.xlabel('Category')
#plt.show()

vectorizer = Vectorizer(train_texts, test_texts)

train_features, test_features = vectorizer.get_vectorized_features(type='tfidf')
linear_svm = LinearSVM(train_features, train_data['Label'], test_data['ID'])
linear_svm.predict_and_save_csv(test_features)