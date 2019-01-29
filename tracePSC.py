#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:25:08 2019

@author: tu
"""

# tracePSC class is designed for generating PSC traces and baseline traces from a neuron voltage clamp recording, given the recording abf file and an excel file with the timing of each detected PSC
# a new tracePSC class is initialized as tracePSC(inputDir, resultDir, sheet, PSC_duration, posOut, negOut)
# the inputDir specifies the path for recording abf file , resultDir specifies the path for the PSC detection result Excel file, sheet specify which sheet of the Excel file contains the PSC timing data, PSC_duration specifies how long do will the detected PSC trace and baseline traces be (default =200),  posOut specifies the directory for saving all the PSC traces, negOut directory specifies the directory for saving all the baseline traces.
import pyabf
import jsonpickle
import matplotlib.pyplot as plt
import matplotlib
class tracePSC:
	def __init__(self, inputDir, resultDir, sheet = 0, PSC_duration = 200, posOut = None, negOut=None):
		self.inputDir = inputDir
		self.posOut = posOut
		self.negOut = negOut
		self.resultDir = resultDir
		self.sheet = sheet if sheet is not None else 0
		self.PSC_duration = int(PSC_duration if PSC_duration is not None else 200)
		print('create variable success!')
    
	# Function readABF: read data from abf file. Implement by pyabf.   
	def readABF(self):
		
		abf = pyabf.ABF(self.inputDir)
		signal = abf.sweepY
		time = abf.sweepX # in second
		return(signal,time)
    
	# Function plotTrace: plot traces from data with given trace and time, used for generating graphs for all detected PSC traces or baseline traces
	def plotTrace(self, event_timing, event_trace):
		
		matplotlib.use('Agg')
		num_of_event = len(event_timing)
		nrows = (num_of_event/5) +1
		ncols = 5
		# all traces will be displayed on a single figure, with 5 columns at each row
		fig = plt.figure(figsize=(20,4*nrows))
		for i in range (0,num_of_event):
			 ax = fig.add_subplot(nrows, ncols, i+1)
			 ax.plot(event_timing[i], event_trace[i], label =str(i))
			 ax.set_xlabel('Time (ms)')
			 ax.set_ylabel('Voltage')
			 ax.legend(loc = 'lower right')
		fig_name = '.' + self.inputDir.split('.')[1]
		# figure will be save in the same folder with abf files
		plt.savefig(fig_name, dpi = 100, facecolor='w', edgecolor='w')        
		#plt.show() # don't show the figure in the IDE since the figure can get very large
		
	# Function getPSCs: read out PSC tracs from abf file and excel file for the timing of PSC
	# return PSC_trace (contains recording signals for all PSC) and PSC_timing (contains the matching time series for each PSC)
	# both PSC_trace and PSC_timing are lists of numpy array(n,) 
	def getPSCs(self):
		import pandas as pd
		signalSeq,timeSeq = self.readABF()
		delta = timeSeq[1]-timeSeq[0] # the delta between each sampling, in second
		# set 10 ms rise range before the PSC timing, and PSC_duration range after the PSC timing to include the entire PSC trace in the detection
		riseTime = int(10/(delta*1000))
		decayTime = int(self.PSC_duration/(delta*1000))
		if self.sheet == 0:
			print("Did not specify the sheet in the result excel file, read the first sheet instead.")
		else:
			worksheet = pd.read_excel(self.resultDir, sheet_name= self.sheet)
			eventTime = worksheet['Time (ms)'].dropna().to_frame()
			num_of_PSC = eventTime.shape[0]
			PSC_trace = [[]]*num_of_PSC
			PSC_timing = [[]]*num_of_PSC
			count = 0
			for eachRow in eventTime.iterrows():
				time = eachRow[1]['Time (ms)']
				index = int(time/(delta*1000))
				PSC_trace[count] = signalSeq[index-riseTime:index+decayTime] # each PSC is presented as 210 ms long of trace by default
				PSC_timing[count] = timeSeq[index-riseTime:index+decayTime]*1000 # convert to ms, the time of each PSC
				count = count+1
			return PSC_trace, PSC_timing
     
	# Function getBaseline: read out basline tracs from abf file and excel file for the timing of baseline
	# baseline traces are extracted before and after each PSC
	# return baseline_trace (contains recording signals for baselines) and baseline_timing (contains the matching time series for each baseline)
	# both baseline_trace and baseline_timing are lists of numpy array(m,)   
	def getBaseline(self):
		_,PSC_timing = self.getPSCs()
		signalSeq,timeSeq = self.readABF()
		delta = timeSeq[1]-timeSeq[0]
		baseline_len = int(10/(delta*1000)) + int(self.PSC_duration/(delta*1000)) # keep the baseline trace the same length as the PSC trace
		baseline_trace = []
		baseline_timing = []
		num_of_PSC = len(PSC_timing)
		pre_PSC_end = 0
		for i in range(0,num_of_PSC-1):
			thisPSC = PSC_timing[i]
			nextPSC = PSC_timing[i+1]
			if len(nextPSC) and len(thisPSC)>0:
				# select the baseline region after the PSC
				current_PSC_end = int(thisPSC[-1]/(1000*delta)) # converted to index
				next_PSC_start = int(nextPSC[0]/(1000*delta)) # converted to index
				baseline_start = current_PSC_end # the begining of the baseline is set to match the end of this PSC
				baseline_end = baseline_start+baseline_len
				if baseline_end < next_PSC_start: # the end index of the baseline should not be larger than the start index of the next PSC
					baseline_trace.append(signalSeq[baseline_start:baseline_end])
					baseline_timing.append(timeSeq[baseline_start:baseline_end]*1000) #convert to ms
				# select the baseline region before the PSC
				current_PSC_start = int(thisPSC[0]/(1000*delta)) # converted to index
				baseline_end =  current_PSC_start # the end of the baseline is set to match the begining of this PSC
				baseline_start = current_PSC_start - baseline_len
				if baseline_start > pre_PSC_end: # the begining index of the baseline should not be prior to the end index of the last PSC
					baseline_trace.append(signalSeq[baseline_start:baseline_end])
					baseline_timing.append(timeSeq[baseline_start:baseline_end]*1000) #convert to ms
				pre_PSC_end  = current_PSC_end
		return baseline_trace, baseline_timing
	
	# save extracted PSC traces to individual txt file.
	# data are serialized by jsonpickle.encode, and stored in the posOut directory
	def savePSCtoJSON(self):
		
		PSC_trace, _ = self.getPSCs()
		file_name_base = self.inputDir.split('.')[1][11:]
		count = 0;
		for eachPSC in PSC_trace:
			PSC_encoded = jsonpickle.encode(eachPSC)
			file_name = self.posOut + '/' + file_name_base + '_positive' + str(count) + '.txt'
			f = open(file_name, 'w+')
			f.write(PSC_encoded)
			f.close()
			count = count+1

	# save extracted basline traces to individual txt file.
	# data are serialized by jsonpickle.encode, and stored in the negOut directory	
	def saveBaselinetoJSON(self):

		baseline_trace, _ = self.getBaseline()
		file_name_base = self.inputDir.split('.')[1][11:]
		count = 0
		for eachBaseline in baseline_trace:
			baseline_encoded = jsonpickle.encode(eachBaseline)
			file_name = self.negOut + '/' + file_name_base + '_negative' + str(count) + '.txt'
			f = open(file_name, 'w+')
			f.write(baseline_encoded)
			f.close()
			count = count+1
		
		
		
		
		
		
		
		
		
		
		
		
