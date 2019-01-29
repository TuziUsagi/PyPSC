# Readme

## Required package

### Tensorflow 1.12
### jsonpickle
### pandas
### csv
### Tensorflow serving docker (optional)

## File description

### trainPSCModel.py
This file defines the model and training specs of a CNN model for recognizing neuron post-synaptic currents (PSCs). The input function expects a tfrecord file that contains all the data or a csv file that contains the path to the data files.

### createTraceReuslts.py
This is an example file that will generate training data and evaluation data for training the PSC model. It generates PSC trace txt files and baseline trace txt files. This file uses the tracePSC class from tracePSC.py file. It expect neuron PSC recording file (abf format), an Excel file that contains the time of each detected PSC.

### generate_dataset_PSC.py
This file takes the directories that contains the PSC trace files and baseline trace files. With input labels, this file will generate the cvs files for training and evaluation dataset. Each row contains the txt file path, and the label.

### combinetfRecord.py
This file iterate the train dataset csv file and puts all the data and label into a tfrecord. The content of the data from every training set data and all labels are put into one tfrecod. Same for the eval dataset.

### tracePSC.py
This file defines a class named tracePSC to facilitate extracting PSC and baseline traces, save traces data into txt file, and plot traces. 

## How to Generate Input Data for Model Training?

### Step 1: prepare PSC and baseline trace data
Use createTraceReuslts.py to generate data txt file. Example abf file and excel file are given in the /abf_files folder. It will generate trace data stored in the /positive_PSC and /negative_PSC folders as shown for the demonstration.
### Step 2: generate csv files with list for file paths assigned for train_set and eval_set
Use generate_dataset_PSC.py to create the /train_set.csv and /eval_set.csv with the files in the /positive_PSC and /negative_PSC folders. 
### Step 3: generate tfrecord with dataset csv file
Use combinetfRecord.py to create /train_set.tfrecord with all the file path and label information from /train_set.csv.
Use combinetfRecord.py to create /eval_set.tfrecord with all the file path and label information from /eval_set.csv.

## Results
PSC training data example:
![PSC](https://raw.githubusercontent.com/TuziUsagi/CNN-based-face-classifier-in-tensorflow/master/results/otherfacedet1.jpg)

Baseline training data example:
![baseline](https://raw.githubusercontent.com/TuziUsagi/CNN-based-face-classifier-in-tensorflow/master/results/detectResult1.jpg)
