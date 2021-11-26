import os
import pandas as pd
import numpy as np
import json
import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data_loader import WSDMDataset
from model import DLRM
from optimizer import CustomSchedule

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ========================== Create dataset =======================

data_dir = "../wsdm_model_data/"

train_data_path = os.path.join(data_dir, 'add_ori_sparse_feat_train_data.txt')
test_data_path = os.path.join(data_dir, 'add_ori_sparse_feat_test_data.txt')

train_data = pd.read_csv(train_data_path, sep="\t")
test_data = pd.read_csv(test_data_path, sep="\t")

train_data["launch_seq"] = train_data.launch_seq.apply(lambda x: json.loads(x))
train_data["playtime_seq"] = train_data.playtime_seq.apply(lambda x: json.loads(x))
train_data["duration_prefer"] = train_data.duration_prefer.apply(lambda x: json.loads(x))
train_data["interact_prefer"] = train_data.interact_prefer.apply(lambda x: json.loads(x))

data = train_data.sample(frac=1).reset_index(drop=True)

print(data.label.value_counts())

assert len(data.columns) == 23

test_data["launch_seq"] = test_data.launch_seq.apply(lambda x: json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))

assert len(test_data.columns) == 23

my_train_data = WSDMDataset(data.iloc[:-6000])
my_val_data = WSDMDataset(data.iloc[-6000:])
my_test_data = WSDMDataset(test_data)


# ========================= Hyper Parameters =======================

# ========================== Loss & Optimizers =======================
# 定义loss
loss = tf.keras.losses.MeanSquaredError()
# loss = tf.keras.losses.CategoricalCrossentropy()
# 定义优化器
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
d_model = 64
learning_rate = CustomSchedule(d_model)
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# 定义metrics
metrics=["mse"]
# metrics=["accuracy"]

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

# 添加 Epoch 标记信息 (使用 `str.format`)
checkpoint_path = "../wsdm_model_data/save_model/epochs-{epoch:04d}.ckpt"  # 保存模型的路径和名称
# 创建一个保存模型权重的回调
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)

lr_schedule = LearningRateScheduler(scheduler)
callbacks = [checkpoint, lr_schedule]


is_classify = False

# 解析特征
train_dataset = my_train_data.get_tf_dataset(is_classify=is_classify)
val_dataset = my_val_data.get_tf_dataset(is_classify=is_classify)
test_dataset = my_test_data.get_tf_dataset(shuffle=False, is_classify=is_classify)

model = DLRM()

model.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)

history = model.fit(train_dataset, epochs=10,
                    validation_data=val_dataset,
                    validation_steps=30,
                    # callbacks=callbacks
                    )

# model.load_weights('../wsdm_model_data/save_model/epochs-0006.ckpt')

model.save_weights(data_dir + "best_weights.h5")

# prediction = []
# for b in test_dataset:
#     p = model(b[0], training=False)
#     prediction += list(np.squeeze(np.array(p)))

prediction = model.predict(test_dataset, steps=len(test_dataset))

# prediction = np.argmax(prediction, axis=1)


test_data["prediction"] = np.reshape(prediction, -1)
test_data = test_data[["user_id", "prediction"]]


test_data.to_csv(data_dir + "dlrm_submission.csv", index=False, header=False, float_format="%.2f")