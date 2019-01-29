#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 18:10:40 2019

@author: tu
"""
# example of using tracePSC class to generate PSC traces and baseline traces
from tracePSC import tracePSC
#list of abf files 
abf_list = ['091716b_0006a.abf']
# corresponding Excel files that contains the time of PSCs in each abf file
excel_list =['091716b.xlsx']
# the name of the sheets that shows the time of PSCs
sheet_list = ['0006']

for i in range(0, len(abf_list)):
	# abf files and Excel files are stored in the abf_files folder
	# PSC traces data will be put in the positive_PSC and baseline traces data will be put in the negative_PSC folder
	newPSC = tracePSC('./abf_files/'+abf_list[i], './abf_files/'+excel_list[i], sheet_list[i], 200, './positive_PSC', './negative_PSC')
	baseline_trace, baseline_timing = newPSC.getBaseline() # get baseline traces
	PSC_trace, PSC_timing=newPSC.getPSCs() # get PSC traces
	newPSC.savePSCtoJSON() # save PSC trace to JSON format
	newPSC.saveBaselinetoJSON() # save baseline trace to JSON format
	newPSC.plotTrace(PSC_timing, PSC_trace) # make plot for all PSC traces
