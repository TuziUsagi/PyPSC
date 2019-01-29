#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:41:29 2019

@author: tu
"""
# This file reads the datasets based on the csv file and combine them into a 
# single tfrecord file.
import numpy as np
import tensorflow as tf
import jsonpickle as jp
import csv
# CSV file that contains the data file paths and their corresponding labels
csvFileName = 'train_set.csv'
# Output path for tfrecord
tfRecordName = 'train_set.tfrecord'
# Get a tfrecord writer
tfwriter  = tf.python_io.TFRecordWriter(tfRecordName)
# Read the csv file
with open(csvFileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
	# For each row of the csv, open the data file, read the content and write into the tfrecord
    for row in csv_reader:
      filename = row[0]
      label = row[1]
      with open(filename) as dataFile:
        tempData = dataFile.read()
        tempData = jp.decode(tempData)
        if(tempData.shape[0] != 4200):
          continue
      feature_key_value_pair = {
        'data': tf.train.Feature(float_list = tf.train.FloatList(value = tempData)),
        'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [label.encode('utf-8')]))
        }
      features = tf.train.Features(feature = feature_key_value_pair)
      example = tf.train.Example(features = features)
      tfwriter.write(example.SerializeToString())
tfwriter.close()
        
