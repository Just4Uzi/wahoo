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
        self.sparse_col = ['ori_sex', 'ori_age', 'ori_education', 'ori_device_type', 'ori_occupation_status']
        self.sparse_col_nunique = {}
        self.df_feat = self.df[self.feat_col]
        self.sparse_feat = self.df[self.sparse_col]
        for c in self.sparse_col:
            self.sparse_col_nunique[c] = self.sparse_feat[c].nunique()


    def get_tf_dataset(self, shuffle=False, use_batch=True, batch_size=64, is_classify=False, label_num=8):
        launch_seq = np.array(list(self.df['launch_seq'])).astype(np.float32)
        playtime_seq = np.array(list(self.df['playtime_seq'])).astype(np.float32)
        duration_prefer = np.array(list(self.df['duration_prefer'])).astype(np.float32)
        interact_prefer = np.array(list(self.df['interact_prefer'])).astype(np.float32)

        feat = np.array(self.df_feat).astype(np.float32)
        sparse_feat = np.array(self.sparse_feat).astype(np.int64)

        if is_classify:
            label = tf.one_hot(list(self.df['label']), depth=label_num)
        else:
            label = np.array(self.df['label']).astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((
            {'launch_seq': launch_seq, 'playtime_seq': playtime_seq, 'duration_prefer': duration_prefer,
             'interact_prefer': interact_prefer, 'dense_feat': feat, 'sparse_feat':sparse_feat},
            label))

        if shuffle:
            ds = ds.shuffle(buffer_size=self.len)
        if use_batch:
            ds = ds.batch(batch_size)
        return ds