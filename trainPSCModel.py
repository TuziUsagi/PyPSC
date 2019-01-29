#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import jsonpickle as jp
import csv


LIST_OF_LABELS = 'PSC,baseline'.split(',')
NCLASSES = 2
output_dir = './PSCmodelout'
DATA_SIZE = 4200
shift_value = 200
useMemDataset = True

hparams = {'train_data_path_mem':'./train_set.csv','eval_data_path_mem':'./eval_set.csv',
          'train_data_path_disk':'./train_set.tfrecord','eval_data_path_disk':'./eval_set.tfrecord',
           'batch_size': 512, 'augment': True, 'learning_rate':0.005,'train_steps':300000, 
           'batch_norm': True, 'reg_constant': 0.005,
           'ksize1':32, 'nfil1':32, 'ksize2':64, 'nfil2':32, 'ksize3':128, 'nfil3':64, 'ksize4':256, 'nfil4':128,
           'dense': 256,'rate':0.1,'kernel_stride':1}


def cnn_model(inputData, mode, hparams):
  # get hyper parameters
  ksize1 = hparams.get('ksize1', 128)
  ksize2 = hparams.get('ksize2', 128)
  ksize3 = hparams.get('ksize3',128)
  ksize4 = hparams.get('ksize4',128)
  nfil1 = hparams.get('nfil1', 10)
  nfil2 = hparams.get('nfil2', 20)
  nfil3 = hparams.get('nfil3', 20)
  nfil4 = hparams.get('nfil4', 20)
  rate = hparams.get('rate', 0.25)
  dense_unit = hparams.get('dense', 200)
  kernel_stride = hparams.get('kernel_stride', 2)
  
  # l2 regularization
  regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
  # first unit, convolutional layer + relu+ maxpooling
  conv_result = tf.layers.conv1d(inputData, filters=nfil1,
                          kernel_size=ksize1, strides=kernel_stride, 
                          padding='same', activation=tf.nn.relu,
                          kernel_initializer=tf.initializers.random_normal(),
                          use_bias=True,bias_initializer=tf.initializers.random_normal(),
                          kernel_regularizer=regularizer)
  conv_result = tf.layers.max_pooling1d(conv_result,pool_size=2, strides=1)
  # second unit, convolutional layer + relu+ maxpooling
  conv_result = tf.layers.conv1d(conv_result, filters=nfil2,
                          kernel_size=ksize2, strides=kernel_stride, 
                          padding='same', activation=tf.nn.relu,
                          kernel_initializer=tf.initializers.random_normal(),
                          use_bias=True,bias_initializer=tf.initializers.random_normal(),
                          kernel_regularizer=regularizer)
  conv_result = tf.layers.max_pooling1d(conv_result,pool_size=2, strides=2)
  # third unit, convolutional layer + relu+ maxpooling
  conv_result = tf.layers.conv1d(conv_result, filters=nfil3,
                          kernel_size=ksize3, strides=kernel_stride, 
                          padding='same', activation=tf.nn.relu,
                          kernel_initializer=tf.initializers.random_normal(),
                          use_bias=True,bias_initializer=tf.initializers.random_normal(),
                          kernel_regularizer=regularizer)
  conv_result = tf.layers.max_pooling1d(conv_result,pool_size=2, strides=2)
  # fourth unit, convolutional layer + relu+ maxpooling
  conv_result = tf.layers.conv1d(conv_result, filters=nfil4,
                          kernel_size=ksize4, strides=kernel_stride, 
                          padding='same', activation=tf.nn.relu,
                          kernel_initializer=tf.initializers.random_normal(),
                          use_bias=True,bias_initializer=tf.initializers.random_normal(),
                          kernel_regularizer=regularizer)
  conv_result = tf.layers.max_pooling1d(conv_result,pool_size=2, strides=2)
  
  # reshape the conv layers to prepare for connection to the dense layer
  outlen = conv_result.shape[1]*conv_result.shape[2]
  conv_result = tf.reshape(conv_result, [-1, outlen]) # flattened

  #apply batch normalization
  if hparams['batch_norm']:
    conv_result = tf.layers.dense(conv_result, dense_unit, activation=None)
    conv_result = tf.layers.batch_normalization(
        conv_result, training=(mode == tf.estimator.ModeKeys.TRAIN)) #only batchnorm when training
    conv_result = tf.nn.relu(conv_result)
  else:  
    conv_result = tf.layers.dense(conv_result, dense_unit, activation=tf.nn.relu)
  #apply dropout
  conv_result = tf.layers.dropout(conv_result, rate=rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

  ylogits = tf.layers.dense(conv_result, NCLASSES, activation=None)
  
  #apply batch normalization once more
  if hparams['batch_norm']:
    ylogits = tf.layers.batch_normalization(ylogits, training=(mode == tf.estimator.ModeKeys.TRAIN))
  return ylogits, NCLASSES

def read_and_preprocess_with_augment(data):
  return read_and_preprocess(data, augment=True)
    
def read_and_preprocess(data, augment=False):
  inputData = data['data']
  inputData = tf.reshape(inputData, (DATA_SIZE,1))
  if augment:
    shift_step = tf.random.uniform([1], minval = -shift_value, maxval=shift_value, dtype=tf.int32)
    inputData = tf.roll(inputData, shift_step[0], 0)
  inputData = inputData - tf.reduce_mean(inputData) #Remove mean
  inputData = (inputData - tf.reduce_min(inputData))/(tf.reduce_max(inputData) - tf.reduce_min(inputData)) #range: 0~1
  inputData = inputData - 0.5 #range: -0.5~0.5
  inputData = inputData * 2 #range: -1~1
  features={}
  features['data'] = inputData
  return features, data['label']

def serving_input_fn():
    # Note: only handles one image at a time 
    feature_placeholders = {'data': tf.placeholder(tf.float32, shape=(DATA_SIZE,1))}
    inputData = feature_placeholders['data']
    inputData = inputData - tf.reduce_mean(inputData) #Remove mean
    inputData = (inputData - tf.reduce_min(inputData))/(tf.reduce_max(inputData) - tf.reduce_min(inputData)) #range: 0~1
    inputData = inputData - 0.5 #range: -0.5~0.5
    inputData = inputData * 2 #range: -1~1
    inputData = tf.expand_dims(inputData,0) #add batch dim
    features = {}
    features['data'] = inputData
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

	# setup the input function. Tensorflow expects a input function takes no arg, 
	# this is a workaround
def make_input_fn_mem(csv_of_filenames, batch_size, mode, augment=False):
    def _input_fn():
      original_data = {'data':[], 'label':[]}
      with open(csv_of_filenames) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
          filename = row[0]
          label = row[1]
          with open(filename) as dataFile:
            tempData = dataFile.read()
            tempData = jp.decode(tempData)
            if(tempData.shape[0] != DATA_SIZE):
              continue
            original_data['data'].append(tf.constant(tempData, tf.float32))
            original_data['label'].append(label)
        dataset = tf.data.Dataset.from_tensor_slices(original_data)
        if augment: 
            dataset = dataset.map(read_and_preprocess_with_augment)
        else:
            dataset = dataset.map(read_and_preprocess)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this
 
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        #return dataset.make_one_shot_iterator().get_next()
        return dataset
    return _input_fn
  
def make_input_fn_disk(tfrecord, batch_size, mode, augment=False):
    def _input_fn():
      def extract_fn(data_record):
        features = {'data': tf.FixedLenFeature([DATA_SIZE,], tf.float32),'label': tf.FixedLenFeature([1], tf.string)}
        sample = tf.parse_single_example(data_record, features)
        sample['data'] = tf.reshape(sample['data'], (DATA_SIZE,1))
        sample['label'] = sample['label'][0]
        return sample
      dataset = tf.data.TFRecordDataset([tfrecord])
      dataset = dataset.map(extract_fn)
      if augment: 
        dataset = dataset.map(read_and_preprocess_with_augment)
      else:
        dataset = dataset.map(read_and_preprocess)
      if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
      else:
        num_epochs = 1 # end-of-input after this
      dataset = dataset.repeat(num_epochs).batch(batch_size)
      return dataset
    return _input_fn
    
	# Custome estimator
def image_classifier(features, labels, mode, params):
  ylogits, nclasses = cnn_model(features['data'], mode, params)

  probabilities = tf.nn.softmax(ylogits)
  class_int = tf.cast(tf.argmax(probabilities, 1), tf.uint8)
  class_str = tf.gather(LIST_OF_LABELS, tf.cast(class_int, tf.int32))
  
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    #convert string label to int
    labels_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LIST_OF_LABELS))
    labels = labels_table.lookup(labels)
    #Get the loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ylogits, labels=tf.one_hot(labels, nclasses)))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = loss+ params['reg_constant'] * sum(reg_losses)
	
    evalmetrics =  {'accuracy': tf.metrics.accuracy(class_int, labels)}
    if mode == tf.estimator.ModeKeys.TRAIN:
      # put loss into summary during training
      tf.summary.scalar('train_loss',loss)  
      # this is needed for batch normalization, but has no effect otherwise
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
         train_op = tf.contrib.layers.optimize_loss(
             loss, 
             tf.train.get_global_step(),
             learning_rate=params['learning_rate'],
             optimizer="Adam")
    else:
      tf.summary.scalar('eval_loss',loss)
      acc,_ = tf.metrics.accuracy(class_int, labels)
      tf.summary.scalar('accuracy', acc)
      train_op = None
  else:
    loss = None
    train_op = None
    evalmetrics = None
 
  return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"probabilities": probabilities, 
                     "classid": class_int, "class": class_str},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=evalmetrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput(
            {"probabilities": probabilities, "classid": class_int, 
             "class": class_str})}
    )

def train_and_evaluate(output_dir, hparams):
  EVAL_INTERVAL = 600 #every 5 minutes    
  estimator = tf.estimator.Estimator(model_fn = image_classifier,
                                     params = hparams,
                                     config= tf.estimator.RunConfig(
                                         save_checkpoints_secs = EVAL_INTERVAL),
                                     model_dir = output_dir)
  if(useMemDataset):
    train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn_mem(
                                        hparams['train_data_path_mem'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.TRAIN,
                                        augment = hparams['augment']),
                                      max_steps = hparams['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn_mem(
                                        hparams['eval_data_path_mem'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.EVAL),
                                    steps = None,
                                    exporters = exporter,
                                    start_delay_secs = EVAL_INTERVAL,
                                    throttle_secs = EVAL_INTERVAL)
  else:
    train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn_disk(
                                        hparams['train_data_path_disk'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.TRAIN,
                                        augment = hparams['augment']),
                                      max_steps = hparams['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn_disk(
                                        hparams['eval_data_path_disk'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.EVAL),
                                    steps = None,
                                    exporters = exporter,
                                    start_delay_secs = EVAL_INTERVAL,
                                    throttle_secs = EVAL_INTERVAL)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

train_and_evaluate(output_dir,hparams)
