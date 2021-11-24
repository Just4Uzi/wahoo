import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, GRU, LSTM, Concatenate


class Sequence(Model):
    def __init__(self, gru_units=32, hiddien_unit=64):
        """
        Deep&Crossing
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param res_dropout: A scalar. Dropout of resnet.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(Sequence, self).__init__()
        self.launch_gru = GRU(gru_units, return_sequences=True)
        self.playtime_gru = GRU(gru_units, return_sequences=True)
        self.concat = Concatenate(axis=1)
        self.dense_1 = Dense(hiddien_unit)
        self.dense_2 = Dense(1)

    def call(self, inputs):
        launch_seq = inputs['launch_seq'] # N, 32
        playtime_seq = inputs['playtime_seq'] # N, 32
        duration_prefer = inputs['duration_prefer'] # N, 16
        interact_prefer = inputs['interact_prefer'] # N, 11
        feat = inputs['feat'] # N, 11

        launch_seq = tf.expand_dims(launch_seq, axis=2) # N, 32, 1
        playtime_seq = tf.expand_dims(playtime_seq, axis=2) # N, 32, 1

        launch_seq_feat = self.launch_gru(launch_seq) # N, 32, 32
        playtime_seq_feat = self.playtime_gru(playtime_seq) # N, 32, 32

        launch_seq_feat = launch_seq_feat[:, :, 0]
        playtime_seq_feat = playtime_seq_feat[:, :, 0]

        all_feat = self.concat([launch_seq_feat, playtime_seq_feat, duration_prefer, interact_prefer, feat]) # N, 102
        all_feat_fc1 = self.dense_1(all_feat) # N, 64
        all_feat_fc2 = self.dense_2(all_feat_fc1)

        return all_feat_fc2