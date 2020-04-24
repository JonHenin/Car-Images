from pathlib import Path

import pickle
import pandas as pd
from keras.models import load_model


class LoadData():
    def __init__(self):
        self.datapath = Path('./data/')
        self.imagespath = Path('./images/')
        self.outputpath = Path('./output/')

    def load_data_to_df(self, is_train=False):

        # Load Datafiles
        df_annos = pd.read_csv(self.datapath / 'cars_annos.csv')
        df_labels = pd.read_csv(self.datapath / 'cars_meta.csv')

        # Combine files to get label names and paths
        df_annos.drop(['Unnamed: 0'], axis=1, inplace=True)
        df_labels.rename(columns={'Unnamed: 0': 'class'}, inplace=True)
        df_labels['class'] = df_labels['class'] + 1 #Offset for Python index starting at 0
        df = df_annos.merge(df_labels, left_on='class', right_on='class')
        df['fpath'] = './images/' + df['image']

        return df[df.test == 0] if is_train else df[df.test == 1]

    def load_model(self, model_name):
        return load_model(self.outputpath / ('models\\' + model_name))

    def load_labels(self, label_name):
        return pickle.loads(open(self.outputpath / ('labels\\' + label_name), "rb").read())
