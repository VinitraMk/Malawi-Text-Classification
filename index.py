from modules.preprocessing import clean_text
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from models.svc import SVC
from models.linear_svm import LinearSVM

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

vectorizer = CountVectorizer(
    stop_words = 'english',
    preprocessor=clean_text
)

train_features = vectorizer.fit_transform(train_texts)
test_features = vectorizer.transform(test_texts)

#svc = SVC(train_features, train_data['Label'], test_data['ID'])
#svc.predict_and_save_csv(test_features)
linear_svm = LinearSVM(train_features, train_data['Label'], test_data['ID'])
linear_svm.predict_and_save_csv(test_features)
'''
linear_model = LinearSVC(kernel='linear', decision_function_shape='ovo', C=1)
linear_model.fit(train_features, train_data['Label'])
rbf_model = LinearSVC(kernel='rbf', decision_function_shape='ovo', C=1, gamma=1)
rbf_model.fit(train_features, train_data['Label'])
sigmoid_model = LinearSVC(kernel='sigmoid', decision_function_shape='ovo', C=1, gamma=1)
sigmoid_model.fit(train_features, train_data['Label'])
poly_model = LinearSVC(kernel='poly', decision_function_shape='ovo', C=1, gamma=1)
poly_model.fit(train_features, train_data['Label'])
y_preds = model.predict(test_features)
y_ids = test_data['ID']
y_ids = pd.DataFrame(y_ids, columns=['ID'])
y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
predictions = y_ids.join(y_preds_df)
predictions.to_csv('output/linear_svm.csv', index=False)
'''