import os
import pandas as pd
import numpy as np

class MeowDataCleaner(object):
    def __init__(self, h5_directory):
        self.h5_directory = h5_directory

    def clean_and_save_data(self):
        h5_files = [f for f in os.listdir(self.h5_directory) if f.endswith('.h5')]
        for h5_file in h5_files:
            original_h5_file_path = os.path.join(self.h5_directory, h5_file)
            df = pd.read_hdf(original_h5_file_path)

            df = df.replace([np.inf, -np.inf], np.nan)
            df.fillna(df.mean(), inplace=True)
            df.drop_duplicates(inplace=True)
            df.to_hdf(original_h5_file_path, key='cleaned_data', mode='w')