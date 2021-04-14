import pandas as pd
import numpy as np
from config.config import Config
from sklearn import svm
from modules.utils import get_filename

class SVC:
    X = None
    y = None
    test_ids = None
    linear_model = None
    rbf_model = None
    poly_model = None
    sigmoid_model = None

    def __init__(self, X, y, test_ids):
        self.X = X
        self.y = y
        self.test_ids = test_ids

    def train_models(self):
        self.linear_model = svm.SVC(kernel='linear', decision_function_shape='ovo', C=1)
        self.linear_model.fit(self.X, self.y)
        self.rbf_model = svm.SVC(kernel='rbf', decision_function_shape='ovo', C=1, gamma=1)
        self.rbf_model.fit(self.X, self.y)
        self.sigmoid_model = svm.SVC(kernel='sigmoid', decision_function_shape='ovo', C=1, gamma=1)
        self.sigmoid_model.fit(self.X, self.y)
        self.poly_model = svm.SVC(kernel='poly', decision_function_shape='ovo', C=1, gamma=1)
        self.poly_model.fit(self.X, self.y)
        
    def predict_and_save_csv(self, test_features):
        print('Training models...\n')
        self.train_models()
        print('Saving predictions to csv...\n')
        titles = ['linear_svc', 'rbf_svc', 'sigmoid_svc', 'poly_svc']
        output_directory = Config().get_config()['output_directory']
        print(output_directory)
        for i, clf in enumerate((self.linear_model, self.rbf_model, self.sigmoid_model, self.poly_model)):
            y_preds = clf.predict(test_features)
            y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
            y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
            predictions = y_ids.join(y_preds_df)
            predictions.to_csv(f'{output_directory}/{get_filename(titles[i])}.csv', index=False)