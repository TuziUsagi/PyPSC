#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 04:54:09 2019

@author: 
"""
import os
import random
import csv

def getFileList(file_dir, expected_data_size):
	files = os.listdir(file_dir)
	num_files = len(files)
	file_list = []
	# if the number of file is greater than the expected data size, randomly select
	if num_files > expected_data_size:
		rand_files_index = random.sample(range(0,num_files),expected_data_size)
		for idx in rand_files_index:
			file_path = file_dir +'/'+ files[idx]
			file_list.append(file_path)
	else:
		for file in os.listdir(file_dir):              
			file_path = file_dir +'/'+ file
			file_list.append(file_path)
	return file_list
	    
	
def generate_dataset_PSC(positive_file_dir, negative_file_dir, expected_data_size, train_ratio, labels):
	# shuffle the file list to prepare split of train and eval data
	positive_files = getFileList(positive_file_dir, expected_data_size[0])
	shuffled_positive_files = positive_files[:]
	random.shuffle(shuffled_positive_files)
	negative_files = getFileList(negative_file_dir, expected_data_size[1])
	shuffled_negative_files = negative_files[:]
	random.shuffle(shuffled_negative_files)

	num_positive_train = int(expected_data_size[0]*train_ratio)
	num_negative_train = int(expected_data_size[1]*train_ratio)
	

	#create training set
	with open('train_set.csv','w') as csvfile:
		writer = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE)
		for i in range(0,num_positive_train):
			writer.writerow([shuffled_positive_files[i],labels[0]])
		for j in range(0,num_negative_train):
			writer.writerow([shuffled_negative_files[j],labels[1]])
	#create evaluation set
	with open('eval_set.csv','w') as csvfile:
		writer = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE)
		for i in range(num_positive_train,expected_data_size[0]):
			writer.writerow([shuffled_positive_files[i],labels[0]])
		for j in range(num_negative_train,expected_data_size[1]):
			writer.writerow([shuffled_negative_files[j],labels[1]])

	        
generate_dataset_PSC('./positive_PSC', './negative_PSC', [220,320], 0.7, ['PSC','baseline']) # the input arguments with list [220,320] specify the number of PSC and baseline traces use for creating dataset
