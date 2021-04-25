from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from config.config import Config
from modules.utils import get_filename
from modules.model_utils import get_pipeline, get_gdsearch

class RandomForest:

    X = None
    y = None
    test_ids = None
    label_encoder = None
    model = None


    def __init__(self, X, y, test_ids, label_encoder):
        self.X = X
        self.y = y
        self.test_ids = test_ids
        self.label_encoder = label_encoder

    def train_model(self):
        pipeline = get_pipeline(RandomForestClassifier(random_state = 42, n_jobs = -1))
        self.model = get_gdsearch(pipeline, 'RandomForest').fit(self.X, self.y)

    def predict_and_save_csv(self, test_features):
        print('Training Random Forest Model...\n')
        self.train_model()
        print('Saving predictions to csv...\n')
        title = get_filename('random_forest')
        output_directory = Config().get_config()['output_directory']
        y_preds = self.model.predict(test_features)
        y_ids = pd.DataFrame(self.test_ids, columns = ['ID'])
        y_preds_df = pd.DataFrame(y_preds, columns = ['Label_Id'])
        predictions = y_ids.join(y_preds_df)
        predictions = self.label_encoder.decode(predictions)
        predictions.to_csv(f'{output_directory}/{title}.csv', index = False)

