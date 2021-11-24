import os
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf


class WSDMDataset():
    def __init__(self, df):
        super(WSDMDataset, self).__init__()
        self.df = df
        self.len = len(df)
        self.cols = list(set(self.df.columns))
        self.feat_col = list(set(self.df.columns) - set(['user_id', 'end_date', 'label', 'launch_seq', 'playtime_seq',
                                                         'duration_prefer', 'interact_prefer']))
        self.df_feat = self.df[self.feat_col]


    def get_tf_dataset(self, shuffle=True, batch_size=64):
        launch_seq = np.array(list(self.df['launch_seq'])).astype(np.float32)
        playtime_seq = np.array(list(self.df['playtime_seq'])).astype(np.float32)
        duration_prefer = np.array(list(self.df['duration_prefer'])).astype(np.float32)
        interact_prefer = np.array(list(self.df['interact_prefer'])).astype(np.float32)

        feat = np.array(self.df_feat).astype(np.float32)

        label = np.array(self.df['label']).astype(np.float32)


        ds = tf.data.Dataset.from_tensor_slices((
            {'launch_seq': launch_seq, 'playtime_seq': playtime_seq, 'duration_prefer': duration_prefer,
             'interact_prefer':interact_prefer, 'feat': feat},
            label))

        if shuffle:
            ds = ds.shuffle(buffer_size=self.len)
        ds = ds.batch(batch_size)
        return ds