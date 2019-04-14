import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)

CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key'] #replace with your column names
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']] #replace with corresponding default values

#Creating an appropriate input function read_dataset
def read_dataset(filename, mode):
    # Adding CSV decoder function and dataset creation and methods
    def input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop('fare_amount')
            return features, label
        #create list of filenames with glob pattern (i.e. data_file_*.csv) (m:1 mapping, read all files with given name pattern from the disk )
        filenames_dataset = tf.data.Dataset.list_files(filename)
        #Read lines from text files (1:m mapping reads data from the collected on previous step) 
        textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset) 
        #converts data into csv format and return dictionaries of features and label
        dataset = textlines_dataset.map(decode_csv) 
        
        dataset = dataset.shuffle(1000).repeat(15).batch(128)
        return dataset.make_one_shot_iterator().get_next()
    return input_fn
        
  
def get_train_input_fn():
  return read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid_input_fn():
  return read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL)


INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

def add_more_features(feats):
  # Nothing to add (yet!) if you want to add any features to your training modify this, look into TensorFlow_estimator_samples. 
  return feats

feature_cols = add_more_features(INPUT_COLUMNS)

#Training model
tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.LinearRegressor(
      feature_columns = feature_cols, model_dir = OUTDIR)
model.train(input_fn = get_train_input_fn(), steps = 200)

"""
log output
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_tf_random_seed': None, '_log_step_count_steps': 100, '_save_summary_steps': 100, '_keep_checkpoint_max': 5, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f4e883afcc0>, '_keep_checkpoint_every_n_hours': 10000, '_save_checkpoints_secs': 600, '_num_worker_replicas': 1, '_global_id_in_cluster': 0, '_task_id': 0, '_is_chief': True, '_train_distribute': None, '_model_dir': 'taxi_trained', '_evaluation_master': '', '_save_checkpoints_steps': None, '_service': None, '_task_type': 'worker', '_master': '', '_num_ps_replicas': 0, '_session_config': None}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into taxi_trained/model.ckpt.
INFO:tensorflow:step = 1, loss = 25085.623
INFO:tensorflow:global_step/sec: 114.153
INFO:tensorflow:step = 101, loss = 10748.32 (0.881 sec)
INFO:tensorflow:Saving checkpoints for 200 into taxi_trained/model.ckpt.
INFO:tensorflow:Loss for final step: 6616.365.
<tensorflow.python.estimator.canned.linear.LinearRegressor at 0x7f4e883afbe0>
"""

#Evaluating Model
metrics = model.evaluate(input_fn = get_valid_input_fn(), steps = None)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

"""
log output
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-14-11:41:48
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from taxi_trained/model.ckpt-200
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-14-11:41:50
INFO:tensorflow:Saving dict for global step 200: average_loss = 111.58863, global_step = 200, loss = 14219.01
RMSE on dataset = 10.563551902770996
"""