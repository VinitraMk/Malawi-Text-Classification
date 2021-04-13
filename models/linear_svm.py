import pandas as pd
import numpy as np
from config.config import Config
from sklearn.svm import LinearSVC

class LinearSVM:
    X = None
    y = None
    test_ids = None
    model = None

    def __init__(self, X, y, test_ids):
        self.X = X
        self.y = y
        self.test_ids = test_ids

    def train_models(self):
        self.model = LinearSVC(C=1, multi_class='crammer_singer')
        self.model.fit(self.X, self.y)
        
    def predict_and_save_csv(self, test_features):
        print('Training models...\n')
        self.train_models()
        print('Saving predictions to csv...\n')
        title = 'linear_svm_2'
        output_directory = Config().get_config()['output_directory']
        print(output_directory)
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
        predictions = y_ids.join(y_preds_df)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)