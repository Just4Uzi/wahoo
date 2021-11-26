import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, GRU, LSTM, Concatenate
from tensorflow.keras.layers import BatchNormalization


MIN_FLOAT = np.finfo(np.float32).min / 100.0


class MLPLayer(Layer):

    def __init__(self, units_list=None, activation=None,
                 **kwargs):

        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]

        # units_list = [input_shape] + units_list

        self.units_list = units_list
        self.mlp = []
        self.activation = activation

        for i, unit in enumerate(units_list):

            if i != len(units_list) - 1:
                initializer = tf.keras.initializers.TruncatedNormal(mean=0.,
                                                                    stddev=1.0 / math.sqrt(unit))
                dense = Dense(units_list[i], kernel_initializer=initializer, activation='relu')
                self.mlp.append(dense)
                norm = BatchNormalization()
                self.mlp.append(norm)

            else:
                initializer = tf.keras.initializers.TruncatedNormal(mean=0.,
                                                                    stddev=1.0 / math.sqrt(unit))

                if self.activation is not None:
                    dense = Dense(units_list[i], kernel_initializer=initializer, activation='relu')
                else:
                    dense = Dense(units_list[i], kernel_initializer=initializer, activation=None)
                self.mlp.append(dense)

    def call(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs



class DLRM(Model):

    """

    Dot interaction layer

    See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
    section 2.1.3. Sparse activations and dense activations are combined.
    Dot interaction is applied to a batch of input Tensors [e1, ..., e_k] of the
    same dimension and the output is a batch of Tensors with all distinct pairwise
    dot products of the form dot(e_i, e_j) for i <= j if self interaction is
    True, otherwise dot(e_i, e_j) i < j
    """

    def __init__(self,
                 dense_feature_dim=11,
                 sparse_feature_number=5,
                 sparse_feature_dim=16,
                 num_field=5,
                 gru_units = 32,
                 sync_mode=None,
                 self_interaction=False,
                 independent_embedding=True
                 ):

        super(DLRM, self).__init__()
        self.dense_feature_dim = dense_feature_dim
        self.bot_layer_sizes = [512, 256, 64, 16]
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.top_layer_sizes = [512, 256, 1]
        self.num_field = num_field
        self.self_interaction = self_interaction
        self.independent_embedding = independent_embedding

        # {'sex':3, 'age':6, 'education':4, 'device_type':5, 'occupation_status':3}
        self.sparse_col_nunique = [3, 6, 4, 5 ,3]

        self.launch_gru = GRU(gru_units, return_sequences=True)
        self.playtime_gru = GRU(gru_units, return_sequences=True)

        self.bot_mlp = MLPLayer(
            units_list=self.bot_layer_sizes,
            activation='relu'
        )

        # `number_features * (num_features + 1) / 2` if self.interaction is True and
        # `number_features * (num_features - 1) / 2` if self.interaction is False

        self.concat_size = int((num_field + 1) * (num_field + 2) / 2) if self.self_interaction else \
            int(num_field * (num_field + 1) / 2)

        self.top_mlp = MLPLayer(
            units_list=self.top_layer_sizes
        )

        if self.independent_embedding:
            self.embeddings = []
            for c_n in self.sparse_col_nunique:
                self.embeddings.append(
                    Embedding(
                        input_dim = c_n,
                        output_dim = self.sparse_feature_dim,
                        embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
                    )
                )
        else:
            self.embedding = Embedding(
                input_dim = self.sparse_feature_number,
                output_dim = self.sparse_feature_dim,
                embeddings_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
            )


    def call(self, inputs):
        """Performs the interaction operation on the tensors in the list.

        :param inputs: include sparse categorical features, (batch_size, sparse_num_field),
                        and dense features, (batch_size, dense_feature_dim)

        :return: predictions
        """

        # split inputs to sparse inputs and dense inputs

        launch_seq = inputs['launch_seq']  # N, 32
        playtime_seq = inputs['playtime_seq']  # N, 32
        duration_prefer = inputs['duration_prefer']  # N, 16
        interact_prefer = inputs['interact_prefer']  # N, 11

        sparse_feat = inputs['sparse_feat']  # N, 5
        dense_feat = inputs['dense_feat']  # N, 11

        sparse_feat = tf.expand_dims(sparse_feat, axis=2)  # N, 5, 1

        launch_seq = tf.expand_dims(launch_seq, axis=2)  # N, 32, 1
        playtime_seq = tf.expand_dims(playtime_seq, axis=2)  # N, 32, 1

        launch_seq_feat = self.launch_gru(launch_seq)  # N, 32, 32
        playtime_seq_feat = self.playtime_gru(playtime_seq)  # N, 32, 32

        # launch_seq_feat = launch_seq_feat[:, :, 0]  # N, 32
        # playtime_seq_feat = playtime_seq_feat[:, :, 0]  # N, 32

        launch_seq_feat = tf.reduce_mean(launch_seq_feat, 2)  # N, 32
        playtime_seq_feat = tf.reduce_mean(playtime_seq_feat, 2)  # N, 32

        # (batch_size, sparse_feature_dim)

        x = self.bot_mlp(dense_feat)  # N, 16


        batch_size, d = x.shape # d : dense_feature_dim

        sparse_embs = []
        if self.independent_embedding:
            for i, embed_layer in enumerate(self.embeddings):
                emb = embed_layer(sparse_feat[:, i, :])  # N, 1, 16
                emb = tf.reshape(emb, shape=[-1, self.sparse_feature_dim])  # N, 16
                sparse_embs.append(emb)
        else:
            for s_input in sparse_feat:
                emb = self.embedding(s_input)
                emb = tf.reshape(emb, shape=[-1, self.sparse_feature_dim])
                sparse_embs.append(emb)

        # concat dense embedding and sparse embeddings, (batch_size, (sparse_num_field + 1), embedding_size)

        # N, 6, 16
        T = tf.reshape(
            tf.concat(
                sparse_embs + [x], axis=1
            ),
            [-1, self.num_field + 1, d]
        )

        # interact features, select upper-triangular portion
        Z = tf.linalg.matmul(T, tf.transpose(T, perm=[0, 2, 1]))  # N, 6, 6

        Z_upper_part = Z - tf.linalg.band_part(Z, num_lower=-1, num_upper=0)  # just use Z upper triangular part  (N, 6, 6)

        # select Z lower triangular part
        if self.self_interaction:
            Z_lower_part = Z - tf.linalg.band_part(tf.ones_like(Z) * MIN_FLOAT, num_lower=0, num_upper=-1)
        else:
            Z_lower_part = tf.linalg.band_part(tf.ones_like(Z) * MIN_FLOAT, num_lower=-1, num_upper=0)

        Z_flat = Z_upper_part + Z_lower_part

        Z_flat = tf.boolean_mask(Z_flat, tf.math.greater(Z_flat, tf.ones_like(Z_flat) * MIN_FLOAT))

        Z_flat = tf.reshape(Z_flat, [-1, self.concat_size])

        R = tf.concat([x] + [Z_flat] + [launch_seq_feat] + [playtime_seq_feat] + [duration_prefer] + [interact_prefer], axis=1)  # N, 122

        y = self.top_mlp(R)


        return y