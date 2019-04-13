import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

print(tf.__version__)

#your data details goes here
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('./taxi-train.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('./taxi-valid.csv', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('./taxi-test.csv', header = None, names = CSV_COLUMNS)

#input_fn to read the training data
def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

#input_fn to read the validation data
def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

#to predict data prediction_input_fn
def make_prediction_input_fn(df,num):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )

#Create feature columns
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns


#Linear Regression with tf.Estimator framework
tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time replace OUTDIR with your output file's path

# TODO: Train a linear regression model
model = tf.estimator.LinearRegressor(
        feature_columns = make_feature_cols(), model_dir = OUTDIR)

model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10)
)

"""Your output will look similar to this
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_train_distribute': None, '_global_id_in_cluster': 0, '_service': None, '_task_type': 'worker', '_tf_random_seed': None, '_num_ps_replicas': 0, '_save_summary_steps': 100, '_log_step_count_steps': 100, '_save_checkpoints_steps': None, '_num_worker_replicas': 1, '_evaluation_master': '', '_save_checkpoints_secs': 600, '_master': '', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f98674661d0>, '_task_id': 0, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': 'taxi_trained', '_session_config': None}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into taxi_trained/model.ckpt.
INFO:tensorflow:step = 1, loss = 21087.043
INFO:tensorflow:global_step/sec: 215.659
INFO:tensorflow:step = 101, loss = 20878.303 (0.467 sec)
INFO:tensorflow:global_step/sec: 308.134
INFO:tensorflow:step = 201, loss = 15126.457 (0.325 sec)
INFO:tensorflow:global_step/sec: 312.589
INFO:tensorflow:step = 301, loss = 10659.191 (0.319 sec)
INFO:tensorflow:global_step/sec: 265.13
INFO:tensorflow:step = 401, loss = 12315.115 (0.378 sec)
INFO:tensorflow:global_step/sec: 311.146
INFO:tensorflow:step = 501, loss = 5760.46 (0.321 sec)
INFO:tensorflow:global_step/sec: 338.472
INFO:tensorflow:step = 601, loss = 14029.924 (0.296 sec)
INFO:tensorflow:Saving checkpoints for 608 into taxi_trained/model.ckpt.
INFO:tensorflow:Loss for final step: 86.690994.
<tensorflow.python.estimator.canned.linear.LinearRegressor at 0x7f9864e84400>"""

#evaluate your Trained model over evaluation data
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, df_valid)

""" 
log output
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-13-17:58:34
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from taxi_trained/model.ckpt-608
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-13-17:58:34
INFO:tensorflow:Saving dict for global step 608: average_loss = 109.4818, global_step = 608, loss = 13020.514
RMSE on dataset = 10.46335506439209
"""

#Predict from the estimator model we trained using test dataset
import itertools
model = tf.estimator.LinearRegressor(
        feature_columns = make_feature_cols(),model_dir = OUTDIR)
preds_iter = model.predict(input_fn = make_prediction_input_fn(df_test,1))
print ([pred['predictions'][0] for pred in list(itertools.islice(preds_iter,5))]) 


"""
log output
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_train_distribute': None, '_global_id_in_cluster': 0, '_service': None, '_task_type': 'worker', '_tf_random_seed': None, '_num_ps_replicas': 0, '_save_summary_steps': 100, '_log_step_count_steps': 100, '_save_checkpoints_steps': None, '_num_worker_replicas': 1, '_evaluation_master': '', '_save_checkpoints_secs': 600, '_master': '', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f98645c8668>, '_task_id': 0, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': 'taxi_trained', '_session_config': None}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from taxi_trained/model.ckpt-608
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
[10.809141, 10.807622, 10.811043, 10.808721, 10.871197]
"""

#Deep Neural Network regression
tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

#Train a DNN regression model
model = tf.estimator.DNNRegressor(
        hidden_units = [32,8,2],# 32 neurons on first layer, 8 on second and 2 on last hidden layer
        feature_columns = make_feature_cols(), model_dir = OUTDIR)

model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10)
)
print_rmse(model,df_valid)

"""
log output
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_train_distribute': None, '_global_id_in_cluster': 0, '_service': None, '_task_type': 'worker', '_tf_random_seed': None, '_num_ps_replicas': 0, '_save_summary_steps': 100, '_log_step_count_steps': 100, '_save_checkpoints_steps': None, '_num_worker_replicas': 1, '_evaluation_master': '', '_save_checkpoints_secs': 600, '_master': '', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f9864ec8c88>, '_task_id': 0, '_keep_checkpoint_every_n_hours': 10000, '_model_dir': 'taxi_trained', '_session_config': None}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into taxi_trained/model.ckpt.
INFO:tensorflow:step = 1, loss = 100201.55
INFO:tensorflow:global_step/sec: 233.734
INFO:tensorflow:step = 101, loss = 9142.674 (0.434 sec)
INFO:tensorflow:global_step/sec: 228.242
INFO:tensorflow:step = 201, loss = 8032.4062 (0.436 sec)
INFO:tensorflow:global_step/sec: 302.321
INFO:tensorflow:step = 301, loss = 6161.755 (0.331 sec)
INFO:tensorflow:global_step/sec: 264.592
INFO:tensorflow:step = 401, loss = 6750.6553 (0.390 sec)
INFO:tensorflow:global_step/sec: 283.86
INFO:tensorflow:step = 501, loss = 7077.872 (0.341 sec)
INFO:tensorflow:global_step/sec: 288.277
INFO:tensorflow:step = 601, loss = 9467.258 (0.346 sec)
INFO:tensorflow:Saving checkpoints for 608 into taxi_trained/model.ckpt.
INFO:tensorflow:Loss for final step: 2227.408.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-13-18:08:14
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from taxi_trained/model.ckpt-608
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-13-18:08:14
INFO:tensorflow:Saving dict for global step 608: average_loss = 109.28457, global_step = 608, loss = 12997.058
RMSE on dataset = 10.453926086425781
"""