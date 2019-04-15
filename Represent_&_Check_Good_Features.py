import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#Loading data set, replace with the path of your dataset
df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")

#peeking the data
df.head()

"""
	longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
0	-114.3	        34.2	    	15.0	            	5612.0	    	1283.0	        1015.0	   	472.0	    	1.5	        66900.0
1	-114.5		34.4		19.0			7650.0		1901.0		1129.0		463.0		1.8		80100.0
2	-114.6		33.7		17.0			720.0		174.0		333.0		117.0		1.7		85700.0
3	-114.6		33.6		14.0			1501.0		337.0		515.0		226.0		3.2		73400.0
4	-114.6		33.6		20.0			1454.0		326.0		624.0		262.0		1.9		65500.0

"""

df.describe()

"""

		longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
count		17000.0		17000.0		17000.0			17000.0		17000.0		17000.0		17000.0		17000.0		17000.0
mean		-119.6		35.6		28.6			2643.7		539.4		1429.6		501.2		3.9		207300.9
std		2.0		2.1		12.6			2179.9		421.5		1147.9		384.5		1.9		115983.8
min		-124.3		32.5		1.0			2.0		1.0		3.0		1.0		0.5		14999.0
25%		-121.8		33.9		18.0			1462.0		297.0		790.0		282.0		2.6		119400.0
50%		-118.5		34.2		29.0			2127.0		434.0		1167.0		409.0		3.5		180400.0
75%		-118.0		37.7		37.0			3151.2		648.2		1721.0		605.2		4.8		265000.0
max		-114.3		42.0		52.0			37937.0		6445.0		35682.0		6082.0		15.0		500001.0

"""

#spliting data for training and evaluation
np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df)) < 0.8 #it splits 80% of random data from total dataset
traindf = df[msk]
evaldf = df[~msk]


#apart from the original features in our dataset we create two new features
#mean values of total_rooms and total_bedrooms are looking freaky. Is that possible any house has 2643 rooms. 
#so what we do, we normalize them by dividing by total households 
def add_more_features(df):
  #Add more features to the dataframe
  df['num_rooms'] = df['total_rooms']/df['households']
  df['num_bedrooms'] = df['total_bedrooms']/df['households']
  return df

# Creating pandas input function
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = add_more_features(df),
    y = df['median_house_value'] / 100000, 
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )


# Defining your feature columns
def create_feature_cols():
  return [
    tf.feature_column.numeric_column('housing_median_age'),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), boundaries = np.arange(32.0,42,1).tolist()),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'), boundaries = np.arange(-124,-114,1).tolist()),
    tf.feature_column.numeric_column('num_rooms'),
    tf.feature_column.numeric_column('num_bedrooms'),
    tf.feature_column.numeric_column('median_income')
    #longitude and latitude data has no meaningful magnitudes. To make it meaningful ve veactorize them using one-hot method. By bucketing them.
  ]

# Creating estimator train and evaluate function
def train_and_evaluate(output_dir, num_train_steps):
  # Create tf.estimator.LinearRegressor, train_spec, eval_spec, and train_and_evaluate using your feature columns
  estimator = tf.estimator.LinearRegressor(model_dir = output_dir, feature_columns = create_feature_cols())
  train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(traindf, 8), max_steps = num_train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn(evaldf, 1),
                                   steps = None,
                                   start_delay_secs = 1, #start evaluting after N seconds
                                   throttle_secs = 10) #evaluate every N second
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Running the model
#Compare the loss by commenting out features at create_feature_cols in different ways
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
train_and_evaluate(OUTDIR, 2000)

"""
log output
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_task_type': 'worker', '_train_distribute': None, '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f7693b97b10>, '_evaluation_master': '', '_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000, '_service': None, '_num_ps_replicas': 0, '_tf_random_seed': None, '_master': '', '_num_worker_replicas': 1, '_task_id': 0, '_log_step_count_steps': 100, '_model_dir': './trained_model', '_global_id_in_cluster': 0, '_save_summary_steps': 100}
/usr/local/envs/py2env/lib/python2.7/site-packages/ipykernel/__main__.py:3: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  app.launch_new_instance()
/usr/local/envs/py2env/lib/python2.7/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
INFO:tensorflow:Running training and evaluation locally (non-distributed).
INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after 10 secs (eval_spec.throttle_secs) or training is finished.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into ./trained_model/model.ckpt.
INFO:tensorflow:loss = 1023.3833, step = 1
INFO:tensorflow:global_step/sec: 102.173
INFO:tensorflow:loss = 48.161934, step = 101 (0.983 sec)
INFO:tensorflow:global_step/sec: 196.966
INFO:tensorflow:loss = 39.196983, step = 201 (0.506 sec)
INFO:tensorflow:global_step/sec: 225.023
INFO:tensorflow:loss = 61.09036, step = 301 (0.444 sec)
INFO:tensorflow:global_step/sec: 163.034
INFO:tensorflow:loss = 71.88446, step = 401 (0.614 sec)
INFO:tensorflow:global_step/sec: 239.296
INFO:tensorflow:loss = 92.69409, step = 501 (0.417 sec)
INFO:tensorflow:global_step/sec: 206.018
INFO:tensorflow:loss = 126.46272, step = 601 (0.488 sec)
INFO:tensorflow:global_step/sec: 229.397
INFO:tensorflow:loss = 73.81889, step = 701 (0.433 sec)
INFO:tensorflow:global_step/sec: 218.726
INFO:tensorflow:loss = 63.241405, step = 801 (0.458 sec)
INFO:tensorflow:Saving checkpoints for 851 into ./trained_model/model.ckpt.
INFO:tensorflow:Loss for final step: 26.572403.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-15-18:51:57
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./trained_model/model.ckpt-851
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-15-18:51:57
INFO:tensorflow:Saving dict for global step 851: average_loss = 0.55321175, global_step = 851, loss = 69.41783
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./trained_model/model.ckpt-851
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 852 into ./trained_model/model.ckpt.
INFO:tensorflow:loss = 72.732574, step = 852
INFO:tensorflow:global_step/sec: 97.8158
INFO:tensorflow:loss = 58.498295, step = 952 (1.027 sec)
INFO:tensorflow:global_step/sec: 203.279
INFO:tensorflow:loss = 41.98052, step = 1052 (0.488 sec)
INFO:tensorflow:global_step/sec: 205.22
INFO:tensorflow:loss = 106.069244, step = 1152 (0.488 sec)
INFO:tensorflow:global_step/sec: 231.506
INFO:tensorflow:loss = 51.585785, step = 1252 (0.432 sec)
INFO:tensorflow:global_step/sec: 186.618
INFO:tensorflow:loss = 48.559837, step = 1352 (0.536 sec)
INFO:tensorflow:global_step/sec: 232.109
INFO:tensorflow:loss = 30.97048, step = 1452 (0.431 sec)
INFO:tensorflow:global_step/sec: 196.632
INFO:tensorflow:loss = 42.881695, step = 1552 (0.509 sec)
INFO:tensorflow:global_step/sec: 231.018
INFO:tensorflow:loss = 71.85047, step = 1652 (0.433 sec)
INFO:tensorflow:Saving checkpoints for 1702 into ./trained_model/model.ckpt.
INFO:tensorflow:Loss for final step: 43.018524.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-15-18:52:04
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./trained_model/model.ckpt-1702
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-15-18:52:05
INFO:tensorflow:Saving dict for global step 1702: average_loss = 0.53941625, global_step = 1702, loss = 67.68675
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./trained_model/model.ckpt-1702
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1703 into ./trained_model/model.ckpt.
INFO:tensorflow:loss = 97.29556, step = 1703
INFO:tensorflow:global_step/sec: 95.3468
INFO:tensorflow:loss = 77.14557, step = 1803 (1.054 sec)
INFO:tensorflow:global_step/sec: 198.665
INFO:tensorflow:loss = 58.1537, step = 1903 (0.502 sec)
INFO:tensorflow:Saving checkpoints for 2000 into ./trained_model/model.ckpt.
INFO:tensorflow:Loss for final step: 65.67714.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-04-15-18:52:10
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./trained_model/model.ckpt-2000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-04-15-18:52:10
INFO:tensorflow:Saving dict for global step 2000: average_loss = 0.5596839, global_step = 2000, loss = 70.229965
""""
