import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

class MeowModel(object):
    def __init__(self, cacheDir):
        if not os.path.exists(cacheDir):
            os.makedirs(cacheDir)
        self.model_path = os.path.join(cacheDir, 'model.joblib')
        self.model = None
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def fit(self, xdf, ydf):
        params = {
            'num_leaves': [31],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [60],
            'feature_fraction': [1.0],
            'force_row_wise': [True],
            'boosting_type': ["gbdt"],

        }

        lgb_model = lgb.LGBMRegressor()
        grid_search = GridSearchCV(estimator=lgb_model, param_grid=params, cv=5, scoring='r2')
        grid_search.fit(xdf, ydf)

        print("Best parameters found: ", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        joblib.dump(self.model, self.model_path)

    def predict(self, xdf):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(xdf)
