import os
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data_loader import WSDMDataset
from model import Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ========================== Create dataset =======================

data_dir = "../wsdm_model_data/"

train_data_path = os.path.join(data_dir, 'train_data.txt')
test_data_path = os.path.join(data_dir, 'test_data.txt')

train_data = pd.read_csv(train_data_path, sep="\t")
test_data = pd.read_csv(test_data_path, sep="\t")

train_data["launch_seq"] = train_data.launch_seq.apply(lambda x: json.loads(x))
train_data["playtime_seq"] = train_data.playtime_seq.apply(lambda x: json.loads(x))
train_data["duration_prefer"] = train_data.duration_prefer.apply(lambda x: json.loads(x))
train_data["interact_prefer"] = train_data.interact_prefer.apply(lambda x: json.loads(x))

data = train_data.sample(frac=1).reset_index(drop=True)

assert len(data.columns) == 18

test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))

assert len(test_data.columns) == 18

my_train_data = WSDMDataset(data.iloc[:-6000])
my_val_data = WSDMDataset(data.iloc[-6000:])
my_test_data = WSDMDataset(test_data)

# 解析特征
train_dataset = my_train_data.get_tf_dataset()
val_dataset = my_val_data.get_tf_dataset()
test_dataset = my_test_data.get_tf_dataset(shuffle=False)

# ========================= Hyper Parameters =======================

# ========================== Loss & Optimizers =======================
# 定义loss
mse = tf.keras.losses.MeanSquaredError()
# 定义优化器
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

def scheduler(epoch):
    thred = 10
    if epoch < thred:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (thred - epoch))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.0001,
    restore_best_weights=True
)

lr_schedule = LearningRateScheduler(scheduler)
callbacks = [early_stopping, lr_schedule]


model = Sequence()

model.compile(loss=mse,
              optimizer=opt,
              metrics=["mse"])

history = model.fit(train_dataset, epochs=5,
                    validation_data=val_dataset,
                    validation_steps=30,
                    callbacks=callbacks)


model.save_weights(data_dir + "best_weights.h5")

prediction = model.predict(test_dataset, steps=len(test_dataset))

test_data["prediction"] = np.reshape(prediction, -1)
test_data = test_data[["user_id", "prediction"]]

test_data.to_csv(data_dir + "sequence_submission.csv", index=False, header=False, float_format="%.2f")