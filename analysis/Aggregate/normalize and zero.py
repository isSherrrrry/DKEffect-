#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:05:29 2019

@author: emilywall
"""

import os
from os import listdir
from os.path import isfile, join
import numpy
import csv

directory = './Aggregate/'

def normalize_all(): 
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    #files = ["markov2x2_repetitions_removed.csv", "markov3x3_repetitions_removed.csv", "markov4x4_repetitions_removed.csv"]
    #for this line, comment out trimming line in else:
    
    for i in range(0, len(files)):
        if (files[i][0] == '.'): 
            continue
        else: 
            print '**processing ', files[i]
            result = numpy.array(list(csv.reader(open(directory + files[i], "rb"), delimiter = ",")))
            result = result[1:, 1:]
            result = result.astype(float)
            normalized = numpy.zeros_like(result)
            for j in range(0, len(result)):
                row_sum = sum(result[j])
                for k in range(0, len(result[j])):
                    normalized[j, k] = result[j, k] / row_sum
            
            with open(directory + "normalized_" + files[i], "w+") as new_file:
                csvWriter = csv.writer(new_file, delimiter = ',')
                csvWriter.writerows(normalized)
                
def zero_diag():
    files = ["t1_q4_agg.csv", "t1_q9_agg.csv", "t1_q16_agg.csv"]
    
    for i in range(0, len(files)):
        if (files[i][0] == '.'): 
            continue
        else: 
            print '**processing ', files[i]
            result = numpy.array(list(csv.reader(open(directory + files[i], "rb"), delimiter = ",")))
            result = result[1:, 1:]
            result = result.astype(float)
            zerod = numpy.zeros_like(result)
            normalized = numpy.zeros_like(result)
            for j in range(0, len(result)):
                row_sum = sum(result[j]) - result[j, j]
                for k in range(0, len(result[j])):
                    if (k == j):
                        zerod[j, k] = 0
                    else: 
                        zerod[j, k] = result[j, k]
                    normalized[j, k] = zerod[j, k] / row_sum
            
            with open(directory + "zero_normalized_" + files[i], "w+") as new_file:
                csvWriter = csv.writer(new_file, delimiter = ',')
                csvWriter.writerows(normalized)
        
def get_probabilities(): 
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    
    for i in range(0, len(files)):
        if (files[i][0:10] != 'normalized'): 
            continue
        else: 
            print '**processing ', files[i]
            result = numpy.array(list(csv.reader(open(directory + files[i], "rb"), delimiter = ",")))
            result = result.astype(float)
            diagonal_sum = 0
            off_diagonal_sum = 0
            diagonal_count = 0
            off_diagonal_count = 0
            for j in range(0, len(result)):
                for k in range(0, len(result[j])):
                    if (j == k): 
                        diagonal_sum += result[j, k]
                        diagonal_count += 1
                    else: 
                        off_diagonal_sum += result[j, k]
                        off_diagonal_count += 1
                        
            diagonal_average = diagonal_sum / diagonal_count
            off_diagonal_average = off_diagonal_sum / off_diagonal_count
            print '  diagonal average: ', diagonal_average
            print '  off-diagonal average: ', off_diagonal_average
            print '    > likelihood ratio: ', diagonal_average / off_diagonal_average


if __name__ == '__main__': 
    
    #normalize_all()
    
    #zero_diag()
    
    get_probabilities()