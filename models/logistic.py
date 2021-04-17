from sklearn.linear_model import LogisticRegression
from modules.model_utils import get_gdsearch, get_pipeline
from modules.utils import get_filename
from config.config import Config
import pandas as pd

class Logistic:
    X = None
    y = None
    test_ids = None
    model = None

    def __init__(self, X, y, test_ids):
        self.X = X
        self.y = y
        self.test_ids = test_ids

    def get_model(self):
        self.model = LogisticRegression(multi_class='ovr')

    def train_model(self):
        self.get_model()
        pipeline = get_pipeline(self.model)
        self.model = get_gdsearch(pipeline).fit(self.X, self.y)
        print('Best estimator params:',self.model.best_params_)

    def predict_and_save_csv(self, test_features):
        print('Training Logistic Classifier...\n')
        self.train_model()
        print('Saving predictions to csv...\n')
        title = get_filename('logistic')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label'])
        predictions = y_ids.join(y_preds_df)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)



    