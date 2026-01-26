import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from log import log

class MeowModel(object):
    def __init__(self, cacheDir,params=None):
        if params is None:
            params = {
                'units': 150,  # LSTM层中的单元数
                'epochs': 50,  # 训练迭代次数
                'batch_size': 32,  # 每次迭代的样本数量
            }
        self.units = params['units']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(None, 1)))  # 假设输入数据是单变量序列
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit(self, xdf, ydf):
        # 将数据转换为适合LSTM的格式，这里假设xdf是形状为[序列长度, 特征数]的数组
        X_train, X_val, y_train, y_val = train_test_split(xdf, ydf, test_size=0.2, random_state=42)
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=True
        )
        log.inf("Done fitting")

    def predict(self, xdf):
        # 预测时也需要将数据转换为LSTM所需的格式
        return self.model.predict(xdf)