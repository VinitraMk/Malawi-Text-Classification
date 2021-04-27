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
    label_encoder = None

    def __init__(self, X, y, test_ids, label_encoder):
        self.X = X
        self.y = y
        self.test_ids = test_ids
        self.label_encoder = label_encoder

    def get_model(self):
        self.model = LogisticRegression(multi_class='ovr', n_jobs = -1, solver='liblinear', penalty='l2',
                random_state = 42)

    def train_model(self, skip_vectorizer):
        self.get_model()
        pipeline = get_pipeline(self.model, 'count', True)
        if skip_vectorizer:
            pipeline = get_pipeline(self.model)
        self.model = get_gdsearch(pipeline, 'Logistic', skip_vectorizer).fit(self.X, self.y)
        print('Best estimator params:',self.model.best_params_)

    def predict_and_save_csv(self, test_features, skip_vectorizer = False):
        print('Training Logistic Classifier...\n')
        self.train_model(skip_vectorizer)
        print('Saving predictions to csv...\n')
        title = get_filename('logistic')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns=['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns=['Label_Id'])
        predictions = y_ids.join(y_preds_df)
        predictions = self.label_encoder.decode(predictions)
        predictions.to_csv(f'{output_directory}/{title}.csv', index=False)



    
