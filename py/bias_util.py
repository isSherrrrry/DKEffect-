#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script provides utility functions for computing the bias metrics.

Created on Thu Jul 20 20:02:57 2017

@author: emilywall
"""
import csv
import json
import math
import sys
import os
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from pprint import pprint
import seaborn as sns

# define some important variable
color_map = { 'none': '#7f7f7f', 'Un-Assign': '#7f7f7f', 'Point Guard': '#1f77b4', 'Shooting Guard': '#ff7f0e', 'Small Forward': '#2ca02c', 'Power Forward': '#d62728', 'Center': '#9467bd' }
num_to_pos_map = { 0: 'Center', 1: 'Shooting Guard', 2: 'Point Guard', 3: 'Small Forward', 4: 'Power Forward', 5: 'none', 6: 'Un-Assign' }
pos_to_num_map = { 'Center': 0, 'Shooting Guard': 1, 'Point Guard': 2, 'Small Forward': 3, 'Power Forward': 4, 'none': 5, 'Un-Assign': 6 }
pos_names = sorted(pos_to_num_map.items(), key = operator.itemgetter(1))
pos_names = [x[0] for x in pos_names]

int_to_num_map = { 'hover': 0, 'drag': 1, 'double_click': 2, 'click': 3, 'set_attribute_weight_vector_drag': 4, 'set_attribute_weight_vector_calc': 5, 'set_attribute_weight_vector_select': 6, 'category_click': 7, 'category_double_click': 8, 'help_hover': 9 }
interaction_names = sorted(int_to_num_map.items(), key = operator.itemgetter(1))
interaction_names = [x[0] for x in interaction_names]

data_pt_list = [8, 11, 13, 23, 30, 47, 49, 51, 58, 79, 85, 86, 107, 113, 114, 127, 128, 134, 136, 161, 162, 166, 167, 170, 185, 195, 199, 213, 253, 278, 289, 299, 309, 310, 324, 335, 339, 348, 380, 383, 384, 391, 394, 395, 433, 442, 463, 473, 496, 513, 520, 537, 554, 562, 570, 571, 574, 585, 590, 597, 600, 603, 606, 611, 621, 640, 678, 686, 689, 702, 751, 752, 755, 762, 763, 764, 771, 799, 819, 824, 828, 852, 854, 859, 878, 884, 887, 894, 900, 903, 919, 920, 921, 942, 952, 959, 966, 980, 992, 995]
c_list = [47, 114, 127, 136, 166, 299, 310, 324, 348, 383, 384, 394, 463, 473, 597, 603, 611, 621, 852, 903]
pf_list = [23, 107, 113, 128, 161, 170, 335, 520, 562, 606, 678, 686, 762, 819, 824, 854, 878, 959, 980, 995]
sf_list = [85, 185, 199, 278, 289, 380, 395, 537, 574, 585, 590, 640, 689, 751, 755, 764, 828, 884, 887, 894]
sg_list = [8, 11, 13, 30, 49, 58, 162, 167, 253, 339, 391, 442, 554, 570, 763, 900, 919, 921, 952, 992]
pg_list = [51, 79, 86, 134, 195, 213, 309, 433, 496, 513, 571, 600, 702, 752, 771, 799, 859, 920, 942, 966]
attrs_all = ['Player', 'Player Anonymized', 'Team', 'Position', 'Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Field Goals Att.', 'Avg. Field Goals Made', 'Avg. Free Throws Att.', 'Avg. Free Throws Made', 'Avg. Minutes', 'Avg. Personal Fouls', 'Avg. Points', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds', 'Avg. Turnovers', 'Games Played', 'Height (Inches)', 'Weight (Pounds)', '+/-']
attrs = attrs_all[4 : -1]
framed_attrs = ['Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds', 'Height (Inches)', 'Weight (Pounds)']
window_methods = ['all', 'fixed', 'classification_v1', 'classification_v2', 'category_v1', 'category_v2']
window_methods = ['fixed'] #emily uncomment this
window_method = 'all' # options include 'all', 'fixed', 'classification_v1', 'classification_v2', 'category_v1', and 'category_v2'
marks = 'categories' # options include 'classifications' and 'categories'
metric_names = ['data_point_coverage', 'data_point_distribution', 'attribute_coverage', 'attribute_distribution', 'attribute_weight_coverage', 'attribute_weight_distribution']
verbose = False
to_filter = True
hover_thresh = 0.100
num_quantiles = 4

base_dir = '/Users/emilywall/git/cs6795proj/'
sub_dir = 'user_data/'
directory = base_dir + sub_dir
data_directory = base_dir + 'data/'
data_file_name = 'bball_top100_decimal.csv'
plot_directory = '../' + sub_dir + 'plots/' + window_method + '/'

all_participants = [1552329410851, 1552418101507, 1552420439428, 1552486286119, 1552576259263, 1553180428606, 1553388376056, 1553526680653, 1553539458285, 1553786216072, 1553789463748, 1553883152091, 1554476728882, 1555525727429]
cond_role = [1552418101507, 1552486286119, 1553388376056, 1553539458285, 1553786216072, 1553883152091]
cond_size = [1552420439428, 1552576259263, 1553180428606, 1552329410851, 1553526680653, 1553789463748, 1554476728882, 1555525727429]


# read in the original data file
def read_data(directory, file_name):
    dataset = []
    attr_value_map = dict()
    
    with open(directory + file_name, 'rb') as data_file:
        reader = csv.reader(data_file)
        first_line = True
        for row in reader:
            if (not first_line): # don't use the header
                cur_player = bball_player(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21])
                dataset.append(cur_player)
            first_line = False
    
    attrs = dataset[0].get_map().keys()
    attrs.remove('Name')
    
    for i in range(0, len(attrs)):
        cur_attr = attrs[i]
        cur_distr = []
        for j in range(0, len(dataset)):
            cur_val = float(dataset[j].get_map()[cur_attr])
            cur_distr.append(cur_val)
        cur_distr.sort()
        
        attr_value_map[cur_attr] = { 'min': np.amin(cur_distr),
                              'max': np.amax(cur_distr),
                              'mean': np.mean(cur_distr),
                              'variance': np.var(cur_distr),
                              'distribution': cur_distr,
                              'dataType': 'numeric' }
    return dataset, attr_value_map
    
# recreate the set of logs from the file
def recreate_logs(directory, file_name):
    all_logs = json.loads(open(directory + file_name).read())
    filtered_logs = []
    attr_logs = []
    item_logs = []
    help_logs = []
    cat_logs = []

    filtered_hovers = 0
    filtered_clicks = 0
    hover_distr = []
    for i in range(0, len(all_logs)):
        cur_log = all_logs[i]
        if (to_filter):
            if (cur_log['customLogInfo']['eventType'].lower() == 'hover'):
                if (cur_log['customLogInfo']['elapsedTime'] < hover_thresh): 
                    filtered_hovers += 1
                    continue
                else: 
                    hover_distr.append(cur_log['customLogInfo']['elapsedTime'])
                    filtered_logs.append(cur_log)
            #elif (cur_log['customLogInfo']['eventType'].lower() == 'click'):
            #    filtered_clicks += 1
            #    continue
            # commented so we don't filter clicks right now
            else: 
                if ('item' in cur_log['eventName'].lower() or 'attribute' in cur_log['eventName'].lower()):
                    filtered_logs.append(cur_log)
                
        if ('attribute' in cur_log['eventName'].lower()): 
            attr_logs.append(cur_log)
        elif ('item' in cur_log['eventName'].lower()):
            item_logs.append(cur_log)
        elif ('help' in cur_log['eventName'].lower()):
            help_logs.append(cur_log)
        elif ('category' in cur_log['eventName'].lower()):
            cat_logs.append(cur_log)
        else: 
            print '***error: unknown log', cur_log
            
    if (to_filter):
        hover_distr = sorted(hover_distr)
        print 'filtered out ', filtered_hovers, ' hovers less than ', hover_thresh, ' s; ', len(hover_distr), ' hovers remaining'
        print 'filtered out ', filtered_clicks, ' clicks'
        print 'item and attribute logs remaining ', len(item_logs), ', ', len(attr_logs)
        all_logs = filtered_logs
        #print 'distribution', hover_distr
        #sns.distplot(hover_distr, hist=False, rug=True);
    
    #print file_name, ': attribute (', len(attr_logs), ') + item (', len(item_logs), ') + help (', len(help_logs), ') + category (', len(cat_logs), ') = ', len(all_logs), ' total logs'
    return all_logs, attr_logs, item_logs, help_logs, cat_logs 

# get the subset of logs that happen in the given time frame having 
# the given interaction types
#   time arg can be a Date object; returns all logs that occurred since 'time'
#   time arg can be an integer; returns the last 'time' logs
def get_log_subset(logs, time, interaction_types):
    log_subset = []
    if (not isinstance(time, datetime) and math.isnan(time)):
        time = len(logs)

    if (isinstance(time, datetime)):
        for i in range(0, len(logs)):
            cur_log = logs[i]
            cur_time = datetime.strptime(cur_log['eventTimeStamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
            cur_event_type = cur_log['customLogInfo']['eventType']
            if (cur_time >= time and (len(interaction_types) == 0 or cur_event_type in interaction_types)):
                log_subset.append(logs[i])
    else:
        if (time > len(logs)): 
            time = len(logs)
        num_logs = 0
        i = len(logs) - 1
        while (i >= 0 and num_logs < time):
            cur_log = logs[i]
            cur_event_type = cur_log['customLogInfo']['eventType']
            if (len(interaction_types) == 0 or cur_event_type in interaction_types):
                log_subset.append(logs[i])
                num_logs += 1
            i = i - 1

    return log_subset
    
# get the subset of logs based on the windowing method
def get_logs_by_window_method(window_method, all_logs, item_logs, attr_logs, i, rolling_dist, attr_weight_rolling_dist, label_indices, cat_indices, prev_decision):
    attr_log_subset = []
    item_log_subset = []

    '''
    if (window_method == 'fixed' and rolling_dist > 0):
        if (i <= rolling_dist):
            if (i <= len(item_logs)):
                item_log_subset = get_item_log_subset(item_logs[0 : i]) 
        else: 
            if (i <= len(item_logs)):
                item_log_subset = get_item_log_subset(item_logs[i - rolling_dist : i])
                #item_log_subset = item_logs[i - rolling_dist : i]
                
        if (i <= attr_weight_rolling_dist):
            if (i <= len(attr_logs)):
                attr_log_subset = get_attr_log_subset(attr_logs[0 : i])
        else:
            if (i <= len(attr_logs)):
                attr_log_subset = get_attr_log_subset(attr_logs[i - rolling_dist : i])
                #attr_log_subset = attr_logs[i - rolling_dist : i]
    '''
    
    if (window_method == 'fixed' and rolling_dist > 0):
        if (i <= rolling_dist):
            if (i <= len(all_logs)):
                item_log_subset = get_item_log_subset(all_logs[0 : i]) 
                attr_log_subset = get_attr_log_subset(all_logs[0 : i])
        else: 
            if (i <= len(all_logs)):
                item_log_subset = get_item_log_subset(all_logs[i - rolling_dist : i])
                attr_log_subset = get_attr_log_subset(all_logs[i - rolling_dist : i])
                #item_log_subset = item_logs[i - rolling_dist : i]
                
    elif (window_method == 'classification_v1'):
        if (i in label_indices):
            if (prev_decision == i):
                if (is_attr_log(all_logs[i])):
                    attr_log_subset = [all_logs[i]]
                elif (is_item_log(all_logs[i])): 
                    item_log_subset = [all_logs[i]]
            else: 
                item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
                attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
            prev_decision = i
    elif (window_method == 'classification_v2'):
        if (i in label_indices):
            if (is_attr_log(all_logs[i])):
                attr_log_subset = [all_logs[i]]
            elif (is_item_log(all_logs[i])): 
                item_log_subset = [all_logs[i]]
            prev_decision = i
        else: 
            item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
            attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
    elif (window_method == 'category_v1'):
        if (i in cat_indices):
            if (prev_decision == i):
                if (is_attr_log(all_logs[i])):
                    attr_log_subset = [all_logs[i]]
                elif (is_item_log(all_logs[i])): 
                    item_log_subset = [all_logs[i]]
            else: 
                item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
                attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
            prev_decision = i
    elif (window_method == 'category_v2'):
        if (i in cat_indices):
            if (is_attr_log(all_logs[i])):
                attr_log_subset = [all_logs[i]]
            elif (is_item_log(all_logs[i])): 
                item_log_subset = [all_logs[i]]
            prev_decision = i
        else: 
            item_log_subset = get_item_log_subset(all_logs[prev_decision : i])
            attr_log_subset = get_attr_log_subset(all_logs[prev_decision : i])
    else: 
        if (i <= len(item_logs)):
            item_log_subset = item_logs[0 : i]
        if (i <= len(attr_logs)):
            attr_log_subset = attr_logs[0 : i]
    
    return attr_log_subset, item_log_subset, prev_decision

# get only the attribute logs from the subset of logs
def get_attr_log_subset(logs):
    attr_logs = []
    for i in range(0, len(logs)):
        cur_log = logs[i]
        if ('attribute' in cur_log['eventName'].lower()): 
            attr_logs.append(cur_log)
            
    return attr_logs
    
# get only the item logs from the subset of logs
def get_item_log_subset(logs):
    item_logs = []
    for i in range(0, len(logs)):
        cur_log = logs[i]
        if ('item' in cur_log['eventName'].lower()): 
            item_logs.append(cur_log)
            
    return item_logs
    
# determine if the given log is an attribute log
def is_attr_log(log): 
    if ('attribute' in log['eventName'].lower()): 
        return True
    else: 
        return False
        
# determine if the given log is an item log
def is_item_log(log): 
    if ('item' in log['eventName'].lower()): 
        return True
    else: 
        return False
    
# separate out the set of logs by data item
def get_logs_by_item(logs):
    log_subsets = dict()
    
    for i in range(0, len(logs)):
        cur_log = logs[i]
        cur_data = cur_log['dataItem']
        cur_queue = []
        if (cur_data['Name'][7 : 10] in log_subsets.keys()): 
            cur_queue = log_subsets[cur_data['Name'][7 : 10]]
        cur_queue.append(cur_log)
        log_subsets[cur_data['Name'][7 : 10]] = cur_queue

    return log_subsets
  
# get the expected value from the markov chain
def get_markov_expected_value(N, k):
    try: 
        num = math.pow(N, k) - math.pow((N - 1), k)
        denom = math.pow(N, (k - 1))
        return num / float(denom)
    except OverflowError: 
        print '** Warning: overflow computing markov expected value with N = ' + str(N) + ' and k = ' + str(k)
        return float(N)
    
# get the quantile that the given value belongs to
# is_increasing is true if the quantiles are increasing in value from Q1, Q2, ...
def get_quantile(quantile_list, value, is_increasing):
    if (is_increasing):
        for i in range(0, len(quantile_list)):
            quant_val = quantile_list[i]
            if (i == 0):
                if (value <= quant_val):
                    return quant_val
            else:
                if (value <= quant_val and value > quantile_list[i - 1]):
                    return quant_val
        return -1;
    
    else: 
        for i in range(0, len(quantile_list)):
            quant_val = quantile_list[i]
            if (i == 0):
                if (value >= quant_val):
                    return quant_val
            else:
                if (value >= quant_val and value < quantile_list[i - 1]):
                    return quant_val
        return -1;
    
# get the final classification of data points as well as lists of decisions
# classifications is a dictionary where (key, value) = (player id, classification)
# decisions_labels is of the form (index, user classification, actual classification, player id)
# decisions_cat is of the form (index, category)
# decisions_help is of the form (index, 'help')
def get_classifications_and_decisions(all_logs, dataset):
    classification = dict()
    decisions_labels = []
    decisions_cat = []
    decisions_help = []

    for i in range(0, len(all_logs)):
        cur_log = all_logs[i]
            
        info = cur_log['customLogInfo']
        if ('classification' in info):
            cur_player = cur_log['dataItem']['Name'].replace('Player ', '')
            cur_class = info['classification']
            if (cur_class != 'none'):
                actual_class = get_bball_player(dataset, cur_player).get_full_map()['Position']
                classification[cur_player] = cur_class
                decisions_labels.append((i, cur_class, actual_class, cur_player))
        elif ('category' in cur_log['eventName'].lower()):
            decisions_cat.append((i, cur_log['dataItem'].replace('category_', '')))
        elif ('help' in cur_log['eventName'].lower()):
            decisions_help.append((i, 'help'))
            
    return classification, decisions_labels, decisions_cat, decisions_help
    
# forward fill the list to remove default -1 values
def forward_fill(arr):
    last_val = arr[0]
    for i in range(0, len(arr)):
        if (arr[i] == -1): 
            arr[i] = last_val
        else: 
            last_val = arr[i]
    return arr
    
# modify array to remove initial -1's then forward fill through other -1's
def remove_defaults(arr, first_decision):
    if (first_decision > -1):
        arr[0 : first_decision] = [0] * first_decision
    arr = forward_fill(arr)
    return arr

# turn the metrics' information into matrices that can be visualized as heat maps
def get_metric_matrices(directory, file_name, user_id, dataset):
    # get a list of data points to make sure matrices use same indexing
    print 'get_metric_matrices'
    data_pts = []
    quantile_map = dict() # save this for writing to files later
    quantile_map['attribute_coverage'] = dict()
    attribute_weight_quantiles = ['-0.5', '0.0', '0.5', '1.0'] 
    
    for i in range(0, len(dataset)):
        cur_player = dataset[i]
        data_pts.append(cur_player.get_map()['Name'])
    data_pts.sort()
        
    # now go through the metrics
    metric_map = dict()
    bias_logs = json.loads(open(directory + file_name).read())
    
    dataset, attr_value_map = read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = get_classifications_and_decisions(recreate_logs(directory, 'user_' + user_id + '/logs/interactions_' + user_id + '.json')[0], dataset)
    decision_pts = dict()
    for i in range(0, len(decisions_labels)):
        # decisions_labels format: (time_stamp, cur_class, actual_class, cur_player)
        cur_label = decisions_labels[i]
        decision_pts[cur_label[0]] = [cur_label[1], cur_label[2], cur_label[3]]
    
    interaction_types = []
    for i in range(0, len(bias_logs)):
        interaction_types.append(bias_logs[i]['interaction_type'])
        cur_log = bias_logs[i]['bias_metrics']
        
        for metric_type in cur_log:
            cur_metric = cur_log[metric_type]
            if (metric_type not in metric_map):
                if ('data_point' in metric_type):
                    metric_map[metric_type] = []
                else:
                    metric_map[metric_type] = {}
                    
        if ('data_point_coverage' in cur_log): 
            cur_metric = cur_log['data_point_coverage']
            visited_pts = cur_metric['info']['visited']
            cur_iter = [0] * len(data_pts)
            for j in range(0, len(visited_pts)):
                cur_iter[data_pts.index(visited_pts[j])] = 1
            metric_map['data_point_coverage'].append(cur_iter)
        else: 
            if ('data_point_coverage' not in metric_map):
                metric_map['data_point_coverage'] = []
            if (len(metric_map['data_point_coverage']) > 0):
                metric_map['data_point_coverage'].append(metric_map['data_point_coverage'][len(metric_map['data_point_coverage']) - 1])
            else:
                metric_map['data_point_coverage'].append([0] * len(data_pts))
           
        if ('data_point_distribution' in cur_log):
            cur_metric = cur_log['data_point_distribution']
            visited_pts = cur_metric['info']['distribution_vector'].keys()
            cur_iter = [0] * len(data_pts)
            for j in range(0, len(visited_pts)):
                cur_iter[data_pts.index(visited_pts[j])] = cur_metric['info']['distribution_vector'][visited_pts[j]]['observed']
            metric_map['data_point_distribution'].append(cur_iter)
        else: 
            if ('data_point_distribution' not in metric_map):
                metric_map['data_point_distribution'] = []
            if (len(metric_map['data_point_distribution']) > 0):
                metric_map['data_point_distribution'].append(metric_map['data_point_distribution'][len(metric_map['data_point_distribution']) - 1])
            else:
                metric_map['data_point_distribution'].append([0] * len(data_pts))
                
        if ('attribute_coverage' in cur_log):
            cur_metric = cur_log['attribute_coverage']
            attributes = cur_metric['info']['attribute_vector'].keys()
            for j in range(0, len(attributes)):
                quantiles = cur_metric['info']['attribute_vector'][attributes[j]]['quantiles']
                quantiles.sort()
                if (attributes[j] not in quantile_map['attribute_coverage']):
                    quantile_map['attribute_coverage'][attributes[j]] = quantiles
                if (attributes[j] not in metric_map['attribute_coverage']): 
                    metric_map['attribute_coverage'][attributes[j]] = []
                cur_iter = [0] * len(quantiles)
                for k in range(0, len(quantiles)):
                    truth_val = cur_metric['info']['attribute_vector'][attributes[j]]['quantile_coverage'][str(quantiles[k])]
                    if (truth_val == True): 
                        cur_iter[k] = 1
                metric_map['attribute_coverage'][attributes[j]].append(cur_iter)
        else:
            if ('attribute_coverage' not in metric_map):
                metric_map['attribute_coverage'] = {}
            attributes = attrs
            for j in range(0, len(attributes)):
                if (attributes[j] not in metric_map['attribute_coverage']): 
                    metric_map['attribute_coverage'][attributes[j]] = []
                if (len(metric_map['attribute_coverage'][attributes[j]]) > 0):
                    metric_map['attribute_coverage'][attributes[j]].append(metric_map['attribute_coverage'][attributes[j]][len(metric_map['attribute_coverage'][attributes[j]]) - 1])
                else:
                    metric_map['attribute_coverage'][attributes[j]].append([0] * len(attribute_weight_quantiles))
                    
        if ('attribute_distribution' in cur_log):
            cur_metric = cur_log['attribute_distribution']
            attributes = cur_metric['info']['attribute_vector'].keys()
            for j in range(0, len(attributes)):
                # figure out quantiles
                quantiles = []
                if (attributes[j] not in metric_map['attribute_distribution']): 
                    metric_map['attribute_distribution'][attributes[j]] = []
                
                # figure out quantization of attribute
                full_dist = cur_metric['info']['attribute_vector'][attributes[j]]['actual_distribution']
                for k in range(0, num_quantiles):
                    if (k != num_quantiles - 1):
                        quant_val = full_dist[int(math.floor((k + 1) * len(dataset) / num_quantiles) - 1)]
                    else:
                        quant_val = full_dist[len(full_dist) - 1]
                    quantiles.append(quant_val)
                quantization = [0] * len(quantiles)
                
                # figure out distribution of interactions
                int_dist = cur_metric['info']['attribute_vector'][attributes[j]]['interaction_distribution']
                for k in range(0, len(int_dist)):
                    cur_val = int_dist[k]
                    # figure out which quantile it belongs to
                    which_quantile = get_quantile(quantiles, cur_val, True)
                    quantization[quantiles.index(which_quantile)] += 1
                metric_map['attribute_distribution'][attributes[j]].append(quantization)
        else:
            if ('attribute_distribution' not in metric_map):
                metric_map['attribute_distribution'] = {}
            attributes = attrs
            for j in range(0, len(attributes)):
                if (attributes[j] not in metric_map['attribute_distribution']): 
                    metric_map['attribute_distribution'][attributes[j]] = []
                if (len(metric_map['attribute_distribution'][attributes[j]]) > 0):
                    metric_map['attribute_distribution'][attributes[j]].append(metric_map['attribute_distribution'][attributes[j]][len(metric_map['attribute_distribution'][attributes[j]]) - 1])
                else:
                    metric_map['attribute_distribution'][attributes[j]].append([0] * len(attribute_weight_quantiles))
                    
        if ('attribute_weight_coverage' in cur_log):
            cur_metric = cur_log['attribute_weight_coverage']
            attributes = cur_metric['info']['attribute_vector'].keys()
            for j in range(0, len(attributes)):
                quantiles = cur_metric['info']['attribute_vector'][attributes[j]]['quantiles']
                quantiles.sort()
                if (attributes[j] not in metric_map['attribute_weight_coverage']): 
                    metric_map['attribute_weight_coverage'][attributes[j]] = []
                cur_iter = [0] * cur_metric['info']['attribute_vector'][attributes[j]]['number_of_quantiles']
                for k in range(0, len(quantiles)):
                    truth_val = cur_metric['info']['attribute_vector'][attributes[j]]['quantile_coverage'][str(quantiles[k])]
                    if (truth_val == True): 
                        cur_iter[k] = 1
                metric_map['attribute_weight_coverage'][attributes[j]].append(cur_iter)
        else:
            if ('attribute_weight_coverage' not in metric_map):
                metric_map['attribute_weight_coverage'] = {}
            for j in range(0, len(attributes)):
                if (attributes[j] not in metric_map['attribute_weight_coverage']): 
                    metric_map['attribute_weight_coverage'][attributes[j]] = []
                if (len(metric_map['attribute_weight_coverage'][attributes[j]]) > 0):
                    metric_map['attribute_weight_coverage'][attributes[j]].append(metric_map['attribute_weight_coverage'][attributes[j]][len(metric_map['attribute_weight_coverage'][attributes[j]]) - 1])
                else:
                    metric_map['attribute_weight_coverage'][attributes[j]].append([0] * len(attribute_weight_quantiles))
            
        if ('attribute_weight_distribution' in cur_log):
            cur_metric = cur_log['attribute_weight_distribution']
            attributes = cur_metric['info']['attribute_vector'].keys()
            for j in range(0, len(attributes)):
                # figure out quantiles
                quantiles = [-0.5, 0.0, 0.5, 1.0]
                if (attributes[j] not in metric_map['attribute_weight_distribution']): 
                    metric_map['attribute_weight_distribution'][attributes[j]] = []
                
                # figure out distribution of interactions
                quantization = [0] * len(quantiles)
                int_dist = cur_metric['info']['attribute_vector'][attributes[j]]['interaction_distribution']
                for k in range(0, len(int_dist)):
                    cur_val = int_dist[k]
                    # figure out which quantile it belongs to
                    which_quantile = get_quantile(quantiles, cur_val, True)
                    quantization[quantiles.index(which_quantile)] += 1
                metric_map['attribute_weight_distribution'][attributes[j]].append(quantization)
        else:
            if ('attribute_weight_distribution' not in metric_map):
                metric_map['attribute_weight_distribution'] = {}
            for j in range(0, len(attributes)):
                if (attributes[j] not in metric_map['attribute_weight_distribution']): 
                    metric_map['attribute_weight_distribution'][attributes[j]] = []
                if (len(metric_map['attribute_weight_distribution'][attributes[j]]) > 0):
                    metric_map['attribute_weight_distribution'][attributes[j]].append(metric_map['attribute_weight_distribution'][attributes[j]][len(metric_map['attribute_weight_distribution'][attributes[j]]) - 1])
                else:
                    metric_map['attribute_weight_distribution'][attributes[j]].append([0] * len(attribute_weight_quantiles))
        
    # write the metric maps to a bunch of files
    metric_matrices_dir = directory + 'user_' + user_id + '/logs/metric_matrices/'
    if not os.path.exists(metric_matrices_dir):
        os.makedirs(metric_matrices_dir)
    
    for key in metric_map:
        metric_info = metric_map[key]
        
        if ('data_point' in key): 
            # a single file for each data point metric
            f_out = open(metric_matrices_dir + key + '.csv', 'wb')
            writer = csv.writer(f_out, delimiter = ',', quotechar = "'", quoting = csv.QUOTE_MINIMAL)
            first_line = ['"data_point"', '"sorted_num"', '"position_num"', '"position"', '"time_stamp"', '"value"', '"decision"', '"interaction_type"']
            writer.writerow(first_line)
            for i in range(0, len(metric_info)):
                time_stamp = i + 1
                time_stamp_str = '"' + str(time_stamp) + '"'
                int_type = interaction_types[i]
                
                # emily: need to record the actual time stamps?
                if (time_stamp in decision_pts): 
                    cur_decision = decision_pts[time_stamp][0]
                else: 
                    cur_decision = 'none'
                
                for j in range(0, len(data_pts)):
                    data_pt = '"' + str(data_pts[j]) + '"'
                    pos = get_bball_player(dataset, data_pts[j]).get_full_map()['Position']
                    if (pos == 'Center'): 
                        pos_num = 5
                        sorted_num = 20 * (pos_num - 1) + c_list.index(int(data_pts[j])) + 1
                    elif (pos == 'Power Forward'):
                        pos_num = 4
                        sorted_num = 20 * (pos_num - 1) + pf_list.index(int(data_pts[j])) + 1
                    elif (pos == 'Small Forward'):
                        pos_num = 3
                        sorted_num = 20 * (pos_num - 1) + sf_list.index(int(data_pts[j])) + 1
                    elif (pos == 'Shooting Guard'):
                        pos_num = 2
                        sorted_num = 20 * (pos_num - 1) + sg_list.index(int(data_pts[j])) + 1
                    elif (pos == 'Point Guard'):
                        pos_num = 1
                        sorted_num = 20 * (pos_num - 1) + pg_list.index(int(data_pts[j])) + 1
                    else: 
                        print 'ERROR: Undefined Position', pos
                    val = metric_info[i][j]
                    row = [data_pt, sorted_num, pos_num, pos, time_stamp_str, val, cur_decision, int_type]
                    writer.writerow(row)
            f_out.close()
        
        else:
            for attr in metric_info:
                if (attr == 'Rand'): 
                    continue
                f_out = open(metric_matrices_dir + key + '_' + attr + '.csv', 'wb')
                writer = csv.writer(f_out, delimiter = ',', quotechar = "'", quoting = csv.QUOTE_MINIMAL)
                first_line = ['"quantile"', '"time_stamp"', '"value"', '"decision"', '"interaction_type"']
                if ('attribute_weight' in key):
                    quantiles = attribute_weight_quantiles # all attribute weights have same quantization
                else: 
                    quantiles = quantile_map['attribute_coverage'][attr]
                writer.writerow(first_line)
                for i in range(0, len(metric_info[attr])):
                    time_stamp = i + 1
                    time_stamp_str = '"' + str(time_stamp) + '"'
                    int_type = interaction_types[i]
                    
                    if (time_stamp in decision_pts): 
                        cur_decision = decision_pts[time_stamp][0]
                    else: 
                        cur_decision = 'none'
                    
                    for j in range(0, len(metric_info[attr][i])):
                        quantile = '"' + str(j + 1) + ' (' + str(quantiles[j]) + ')"'
                        val = metric_info[attr][i][j]
                        row = [quantile, time_stamp_str, val, cur_decision, int_type]
                        writer.writerow(row)
                f_out.close()
                
    f_out = open(directory + 'user_' + user_id + '/logs/metric_matrix.json', 'w+')
    f_out.write(json.dumps(metric_map))
    f_out.close()
    print '**file written**'
    
    return metric_map
        
# plot the metric over time marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric(x_values, metric_values, title, x_label, y_label, directory, file_name, decisions, marks, fig_num, verbose):
    if (verbose): 
        print 'Plotting', title
        
    plt.figure(num = fig_num, figsize = (15, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sub_plot = plt.subplot(1, 1, 1)
    x_values = np.array(x_values)
    metric_values = np.array(metric_values)
    plt.plot(x_values, metric_values, c = '#000000')
    
    if (marks == 'classifications'):
        for i in range(0, len(decisions)):
            tup = decisions[i]
            if (tup[1] == tup[2]): 
                line_width = 4
            else: 
                line_width = 2
            sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = line_width, zorder = 0, clip_on = False)
    elif (marks == 'categories'):
        for i in range(0, len(decisions)): 
            tup = decisions[i]
            sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
    
    # label, save, and clear
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# plot multiple metrics over time in subplots of one figure marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric_with_subplot(x_values, metric_values, titles, x_label, y_label, directory, file_name, decisions, marks, fig_num, verbose):
    if (verbose): 
        print 'Plotting Subplots'

    plt.figure(num = fig_num, figsize = (15, 60), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(0, len(x_values)):
        cur_x = np.array(x_values[i])
        cur_y = np.array(metric_values[i])
        sub_plot = plt.subplot(len(x_values), 1, i + 1)
        plt.plot(cur_x, cur_y, c = '#000000')
            
        # axis labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(titles[i])
        
        if (marks == 'classifications'):
            for i in range(0, len(decisions)):
                tup = decisions[i]
                if (tup[1] == tup[2]): 
                    line_width = 4
                else: 
                    line_width = 2
                sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = line_width, zorder = 0, clip_on = False)
        elif (marks == 'categories'):
            for i in range(0, len(decisions)): 
                tup = decisions[i]
                sub_plot.axvline(x = tup[0], c = color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
         
    plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# plot the metric over time as a heat map style vis, marking the times where 
# (1) points were classified if marks == 'classifications', or 
# (2) categories were clicked if marks == 'categories' 
def plot_metric_heat_map(matrix, title, x_label, y_label, directory, file_name, fig_num):
    if (verbose):
        print 'Plotting', title
    
    plt.figure(num = fig_num, figsize = (len(matrix), len(matrix[0])), dpi = 40, facecolor = 'w', edgecolor = 'k') #dpi = 60, 
    plt.subplot(1, 1, 1)
    matrix = np.array(matrix)
    heatmap = plt.pcolor(matrix, cmap = 'Blues')
    
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%d' % matrix[y, x], horizontalalignment = 'center', verticalalignment = 'center')
    
    plt.colorbar(heatmap)
    plt.gca().invert_yaxis()
    #plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation = 'vertical')
    #plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    '''plt.xlabel(x_label, fontsize = 200)
    plt.ylabel(y_label, fontsize = 200)
    plt.title(title, fontsize = 200)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)'''
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    font = { 'family': 'normal', 'weight': 'bold', 'size': 200 }
    matplotlib.rc('font', **font)

    #plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# get the 0-1 bias metric values over time for all users and write it to a single file
def write_timeline(ids):
    all_users_bias = dict()
    
    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        
        user_window_methods_bias = dict()
        
        # separate computation for each windowing method
        for j in range(0, len(window_methods)):
            wm = window_methods[j]
            
            # initialize a bunch of dictionaries to store shit
            user_window_methods_bias[wm] = dict()
            
            user_window_methods_bias[wm][metric_names[0]] = []
            user_window_methods_bias[wm][metric_names[1]] = []
            user_window_methods_bias[wm][metric_names[2]] = dict()
            user_window_methods_bias[wm][metric_names[3]] = dict()
            user_window_methods_bias[wm][metric_names[4]] = dict()
            user_window_methods_bias[wm][metric_names[5]] = dict()
            
            # attribute and attribute weight metrics gotta have another layer of depth per attribute
            for k in range(0, len(attrs)):
                ak = attrs[k]
                user_window_methods_bias[wm][metric_names[2]][ak] = []
                user_window_methods_bias[wm][metric_names[3]][ak] = []
                user_window_methods_bias[wm][metric_names[4]][ak] = []
                user_window_methods_bias[wm][metric_names[5]][ak] = []
            
            cur_file_name = directory + 'user_' + cur_user + '/logs/bias_' + wm + '_' + cur_user + '.json'
            cur_file = json.loads(open(cur_file_name).read())
            
            # go through each computed bias value in time
            for k in range(0, len(cur_file)):
                if (metric_names[0] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[0]].append(cur_file[k]['bias_metrics'][metric_names[0]]['metric_level'])
                if (metric_names[1] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[1]].append(cur_file[k]['bias_metrics'][metric_names[1]]['metric_level'])
                for l in range(0, len(attrs)):
                    al = attrs[l]
                    
                    if (metric_names[2] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[2]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[2]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[3] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[3]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[3]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[4] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[4]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[4]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[5] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[5]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[5]]['info']['attribute_vector'][al]['metric_level'])
    
        all_users_bias[cur_user] = user_window_methods_bias
        
    # now write it to a file
    for i in range(0, len(window_methods)):
        wm = window_methods[i]
        
        for j in range(0, len(metric_names)):
            
            # write data point metrics to a file
            if ('data_point' in metric_names[j]):
                out_file_name = directory + 'analysis/' + metric_names[j] + '_' + wm + '_time_series.csv'
                f_out = open(out_file_name, 'wb')
                writer = csv.writer(f_out, delimiter = ',', quotechar = "'", quoting = csv.QUOTE_MINIMAL)
                first_line = ['time_stamp'] + [str(cur_id) for cur_id in all_participants]
                writer.writerow(first_line)
                num_users = len(all_participants)
                time_stamp = 1
                while (num_users > 0):
                    metric_vals = []
                    next_num = 0
                    for k in range(0, len(all_participants)):
                        if (len(all_users_bias[str(all_participants[k])][wm][metric_names[j]]) > 0):
                            metric_vals.append(str(all_users_bias[str(all_participants[k])][wm][metric_names[j]][0]))
                            all_users_bias[str(all_participants[k])][wm][metric_names[j]] = all_users_bias[str(all_participants[k])][wm][metric_names[j]][1:]
                            if (len(all_users_bias[str(all_participants[k])][wm][metric_names[j]]) > 0):
                                next_num += 1
                        else:
                            metric_vals.append('')
                    row = [str(time_stamp)] + metric_vals
                    writer.writerow(row)
                    time_stamp += 1
                    num_users = next_num
                f_out.close()
                
            else:
                for k in range(0, len(attrs)):
                    ak = attrs[k]
                    
                    out_file_name = directory + 'analysis/' + metric_names[j] + '_' + ak + '_' + wm + '_time_series.csv'
                    f_out = open(out_file_name, 'wb')
                    writer = csv.writer(f_out, delimiter = ',', quotechar = "'", quoting = csv.QUOTE_MINIMAL)
                    first_line = ['time_stamp'] + [str(cur_id) for cur_id in all_participants]
                    writer.writerow(first_line)
                    num_users = len(all_participants)
                    time_stamp = 1
                    while (num_users > 0):
                        metric_vals = []
                        next_num = 0
                        for l in range(0, len(all_participants)):
                            if (len(all_users_bias[str(all_participants[l])][wm][metric_names[j]][ak]) > 0):
                                metric_vals.append(str(all_users_bias[str(all_participants[l])][wm][metric_names[j]][ak][0]))
                                all_users_bias[str(all_participants[l])][wm][metric_names[j]][ak] = all_users_bias[str(all_participants[l])][wm][metric_names[j]][ak][1:]
                                if (len(all_users_bias[str(all_participants[l])][wm][metric_names[j]][ak]) > 0):
                                    next_num += 1
                            else:
                                metric_vals.append('')
                        row = [str(time_stamp)] + metric_vals
                        writer.writerow(row)
                        time_stamp += 1
                        num_users = next_num
                    f_out.close()
    
    return all_users_bias

# compute the average / max / last bias metric value over time for the given participants
# comp_type can be 'avg', 'max', or 'last'
def avg_bias_values(ids, comp_type):
    all_users_bias = dict()
    all_users_summary = dict()
    
    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        
        user_window_methods_bias = dict()
        user_window_methods_summary = dict()
        
        # separate computation for each windowing method
        for j in range(0, len(window_methods)):
            wm = window_methods[j]
            
            # initialize a bunch of dictionaries to store shit
            user_window_methods_bias[wm] = dict()
            user_window_methods_summary[wm] = dict()
            
            user_window_methods_bias[wm][metric_names[0]] = []
            user_window_methods_bias[wm][metric_names[1]] = []
            user_window_methods_bias[wm][metric_names[2]] = dict()
            user_window_methods_bias[wm][metric_names[3]] = dict()
            user_window_methods_bias[wm][metric_names[4]] = dict()
            user_window_methods_bias[wm][metric_names[5]] = dict()
            
            # attribute and attribute weight metrics gotta have another layer of depth per attribute
            user_window_methods_summary[wm][metric_names[2]] = dict()
            user_window_methods_summary[wm][metric_names[3]] = dict()
            user_window_methods_summary[wm][metric_names[4]] = dict()
            user_window_methods_summary[wm][metric_names[5]] = dict()
            for k in range(0, len(attrs)):
                ak = attrs[k]
                user_window_methods_bias[wm][metric_names[2]][ak] = []
                user_window_methods_bias[wm][metric_names[3]][ak] = []
                user_window_methods_bias[wm][metric_names[4]][ak] = []
                user_window_methods_bias[wm][metric_names[5]][ak] = []
            
            cur_file_name = directory + 'user_' + cur_user + '/logs/bias_' + wm + '_' + cur_user + '.json'
            cur_file = json.loads(open(cur_file_name).read())
            
            # go through each computed bias value in time
            for k in range(0, len(cur_file)):
                if (metric_names[0] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[0]].append(cur_file[k]['bias_metrics'][metric_names[0]]['metric_level'])
                if (metric_names[1] in cur_file[k]['bias_metrics']):
                    user_window_methods_bias[wm][metric_names[1]].append(cur_file[k]['bias_metrics'][metric_names[1]]['metric_level'])
                for l in range(0, len(attrs)):
                    al = attrs[l]
                    if (metric_names[2] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[2]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[2]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[3] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[3]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[3]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[4] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[4]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[4]]['info']['attribute_vector'][al]['metric_level'])
                    if (metric_names[5] in cur_file[k]['bias_metrics']):
                        user_window_methods_bias[wm][metric_names[5]][attrs[l]].append(cur_file[k]['bias_metrics'][metric_names[5]]['info']['attribute_vector'][al]['metric_level'])
            
            # compute summary: average, max, or last
            if (comp_type == 'avg'):
                user_window_methods_summary[wm][metric_names[0]] = np.mean(user_window_methods_bias[wm][metric_names[0]])
                user_window_methods_summary[wm][metric_names[1]] = np.mean(user_window_methods_bias[wm][metric_names[1]])
                for k in range(0, len(attrs)):
                    ak = attrs[k]
                    user_window_methods_summary[wm][metric_names[2]][ak] = np.mean(user_window_methods_bias[wm][metric_names[2]][ak])
                    user_window_methods_summary[wm][metric_names[3]][ak] = np.mean(user_window_methods_bias[wm][metric_names[3]][ak])
                    user_window_methods_summary[wm][metric_names[4]][ak] = np.mean(user_window_methods_bias[wm][metric_names[4]][ak])
                    user_window_methods_summary[wm][metric_names[5]][ak] = np.mean(user_window_methods_bias[wm][metric_names[5]][ak])
            elif (comp_type == 'max'):
                user_window_methods_summary[wm][metric_names[0]] = np.amax(user_window_methods_bias[wm][metric_names[0]])
                user_window_methods_summary[wm][metric_names[1]] = np.amax(user_window_methods_bias[wm][metric_names[1]])
                for k in range(0, len(attrs)):
                    user_window_methods_summary[wm][metric_names[2]][ak] = np.amax(user_window_methods_bias[wm][metric_names[2]][ak])
                    user_window_methods_summary[wm][metric_names[3]][ak] = np.amax(user_window_methods_bias[wm][metric_names[3]][ak])
                    user_window_methods_summary[wm][metric_names[4]][ak] = np.amax(user_window_methods_bias[wm][metric_names[4]][ak])
                    user_window_methods_summary[wm][metric_names[5]][ak] = np.amax(user_window_methods_bias[wm][metric_names[5]][ak])
            else: # comp_type == 'last'
                user_window_methods_summary[wm][metric_names[0]] = user_window_methods_bias[wm][metric_names[0]][-1]
                user_window_methods_summary[wm][metric_names[1]] = user_window_methods_bias[wm][metric_names[1]][-1]
                for k in range(0, len(attrs)):
                    user_window_methods_summary[wm][metric_names[2]][ak] = user_window_methods_bias[wm][metric_names[2]][ak][-1]
                    user_window_methods_summary[wm][metric_names[3]][ak] = user_window_methods_bias[wm][metric_names[3]][ak][-1]
                    user_window_methods_summary[wm][metric_names[4]][ak] = user_window_methods_bias[wm][metric_names[4]][ak][-1]
                    user_window_methods_summary[wm][metric_names[5]][ak] = user_window_methods_bias[wm][metric_names[5]][ak][-1]
            
        all_users_bias[cur_user] = user_window_methods_bias
        all_users_summary[cur_user] = user_window_methods_summary
    
    # now summarize it for participants in each condition
    results_1 = dict()
    results_2 = dict()
    for i in range(0, len(window_methods)):
        wm = window_methods[i]
        results_1[wm] = dict()
        results_2[wm] = dict()
        
        results_1[wm][metric_names[0]] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[0]], all_users_summary[str(cond_size[1])][wm][metric_names[0]], all_users_summary[str(cond_size[2])][wm][metric_names[0]], all_users_summary[str(cond_size[3])][wm][metric_names[0]], all_users_summary[str(cond_size[4])][wm][metric_names[0]]])
        results_2[wm][metric_names[0]] = np.mean([all_users_summary[str(cond_role[0])][wm][metric_names[0]], all_users_summary[str(cond_role[1])][wm][metric_names[0]], all_users_summary[str(cond_role[2])][wm][metric_names[0]], all_users_summary[str(cond_role[3])][wm][metric_names[0]], all_users_summary[str(cond_role[4])][wm][metric_names[0]]])
        results_1[wm][metric_names[1]] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[1]], all_users_summary[str(cond_size[1])][wm][metric_names[1]], all_users_summary[str(cond_size[2])][wm][metric_names[1]], all_users_summary[str(cond_size[3])][wm][metric_names[1]], all_users_summary[str(cond_size[4])][wm][metric_names[1]]])
        results_2[wm][metric_names[1]] = np.mean([all_users_summary[str(cond_role[0])][wm][metric_names[1]], all_users_summary[str(cond_role[1])][wm][metric_names[1]], all_users_summary[str(cond_role[2])][wm][metric_names[1]], all_users_summary[str(cond_role[3])][wm][metric_names[1]], all_users_summary[str(cond_role[4])][wm][metric_names[1]]])
        
        results_1[wm][metric_names[2]] = dict()
        results_2[wm][metric_names[2]] = dict()
        results_1[wm][metric_names[3]] = dict()
        results_2[wm][metric_names[3]] = dict()
        results_1[wm][metric_names[4]] = dict()
        results_2[wm][metric_names[4]] = dict()
        results_1[wm][metric_names[5]] = dict()
        results_2[wm][metric_names[5]] = dict()
        
        for j in range(0, len(attrs)):
            aj = attrs[j]
            results_1[wm][metric_names[2]][aj] = np.std([all_users_summary[str(cond_size[0])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[2]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[2]][aj]])
            results_2[wm][metric_names[2]][aj] = np.std([all_users_summary[str(cond_role[0])][wm][metric_names[2]][aj], all_users_summary[str(cond_role[1])][wm][metric_names[2]][aj], all_users_summary[str(cond_role[2])][wm][metric_names[2]][aj], all_users_summary[str(cond_role[3])][wm][metric_names[2]][aj], all_users_summary[str(cond_role[4])][wm][metric_names[2]][aj]])
            results_1[wm][metric_names[3]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[3]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[3]][aj]])
            results_2[wm][metric_names[3]][aj] = np.mean([all_users_summary[str(cond_role[0])][wm][metric_names[3]][aj], all_users_summary[str(cond_role[1])][wm][metric_names[3]][aj], all_users_summary[str(cond_role[2])][wm][metric_names[3]][aj], all_users_summary[str(cond_role[3])][wm][metric_names[3]][aj], all_users_summary[str(cond_role[4])][wm][metric_names[3]][aj]])
            results_1[wm][metric_names[4]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[4]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[4]][aj]])
            results_2[wm][metric_names[4]][aj] = np.mean([all_users_summary[str(cond_role[0])][wm][metric_names[4]][aj], all_users_summary[str(cond_role[1])][wm][metric_names[4]][aj], all_users_summary[str(cond_role[2])][wm][metric_names[4]][aj], all_users_summary[str(cond_role[3])][wm][metric_names[4]][aj], all_users_summary[str(cond_role[4])][wm][metric_names[4]][aj]])
            results_1[wm][metric_names[5]][aj] = np.mean([all_users_summary[str(cond_size[0])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[1])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[2])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[3])][wm][metric_names[5]][aj], all_users_summary[str(cond_size[4])][wm][metric_names[5]][aj]])
            results_2[wm][metric_names[5]][aj] = np.mean([all_users_summary[str(cond_role[0])][wm][metric_names[5]][aj], all_users_summary[str(cond_role[1])][wm][metric_names[5]][aj], all_users_summary[str(cond_role[2])][wm][metric_names[5]][aj], all_users_summary[str(cond_role[3])][wm][metric_names[5]][aj], all_users_summary[str(cond_role[4])][wm][metric_names[5]][aj]])
            
    results = dict()
    results['condition_1_size'] = results_1
    results['condition_2_varying'] = results_2
    #print '**results', results
    return results #emily TODO write this to a file # emily TODO debug this for max and last
    
# determine whether the given weight vector is a custom axis or not
def is_custom_axis(weights):
    weight_one = False
    for i in range(0, len(attrs)):
        if ((weights[attrs[i]] != 0 and weights[attrs[i]] != 1) or weights[attrs[i]] == 1 and weight_one == True): # custom axis
            return True
        if (weights[attrs[i]] == 1):
            weight_one = True
    return False
    
# representation of a basketball player
class bball_player:
    def __init__(self, player, player_anon, team, pos, avg_3p_att, avg_3p_made, avg_ast, avg_blks, avg_fg_att, avg_fg_made, avg_ft_att, avg_ft_made, avg_min, avg_pf, avg_pts, avg_or, avg_st, avg_tr, avg_to, games, height, weight):
        self.player = player
        self.player_anon = player_anon
        self.team = team
        self.pos = pos
        self.avg_3p_att = avg_3p_att
        self.avg_3p_made = avg_3p_made
        self.avg_ast = avg_ast
        self.avg_blks = avg_blks
        self.avg_fg_att = avg_fg_att
        self.avg_fg_made = avg_fg_made
        self.avg_ft_att = avg_ft_att
        self.avg_ft_made = avg_ft_made
        self.avg_min = avg_min
        self.avg_pf = avg_pf
        self.avg_pts = avg_pts
        self.avg_or = avg_or
        self.avg_st = avg_st
        self.avg_tr = avg_tr
        self.avg_to = avg_to
        self.games = games
        self.height = height
        self.weight = weight
        
    def get_map(self):
        return {'Avg. 3-Pointers Att.': self.avg_3p_att, 'Avg. 3-Pointers Made': self.avg_3p_made, 'Avg. Assists': self.avg_ast, 'Avg. Blocks': self.avg_blks, 'Avg. Field Goals Att.': self.avg_fg_att, 'Avg. Field Goals Made': self.avg_fg_made, 'Avg. Free Throws Att.': self.avg_ft_att, 'Avg. Free Throws Made': self.avg_ft_made, 'Avg. Minutes': self.avg_min, 'Avg. Personal Fouls': self.avg_pf, 'Avg. Points': self.avg_pts, 'Avg. Offensive Rebounds': self.avg_or, 'Avg. Steals': self.avg_st, 'Avg. Total Rebounds': self.avg_tr, 'Avg. Turnovers': self.avg_to, 'Games Played': self.games, 'Height (Inches)': self.height, 'Weight (Pounds)': self.weight, 'Name': self.player_anon}
                
    def get_full_map(self):
        return {'Avg. 3-Pointers Att.': self.avg_3p_att, 'Avg. 3-Pointers Made': self.avg_3p_made, 'Avg. Assists': self.avg_ast, 'Avg. Blocks': self.avg_blks, 'Avg. Field Goals Att.': self.avg_fg_att, 'Avg. Field Goals Made': self.avg_fg_made, 'Avg. Free Throws Att.': self.avg_ft_att, 'Avg. Free Throws Made': self.avg_ft_made, 'Avg. Minutes': self.avg_min, 'Avg. Personal Fouls': self.avg_pf, 'Avg. Points': self.avg_pts, 'Avg. Offensive Rebounds': self.avg_or, 'Avg. Steals': self.avg_st, 'Avg. Total Rebounds': self.avg_tr, 'Avg. Turnovers': self.avg_to, 'Games Played': self.games, 'Height (Inches)': self.height, 'Weight (Pounds)': self.weight, 'Name': self.player_anon, 'Team': self.team, 'Position': self.pos, 'Name (Real)': self.player}
    
# get the bball player by the name attribute
def get_bball_player(players, name):
    for i in range(0, len(players)):
        if (players[i].get_map()['Name'] == name): 
            return players[i]

    print '*** Unable to locate player', name
    return -1

# this block is for testing util functions
if __name__ == '__main__':
    
    print directory
    dataset, attr_value_map = read_data('/Users/emilywall/git/bias_eval/data/', 'bball_top100_decimal.csv')
    #test_id = str(1506629987658)
    #test_id = str(1508339795840)
    #test_id = str(1509482115747)
    
    #all_participants = [str(1507820577674)]
    all_participants = [str(1506460542091)]
    
    for i in range(0, len(all_participants)):
        cur_id = str(all_participants[i])
        print '****', cur_id
        metric_matrix = get_metric_matrices(directory, 'user_' + cur_id + '/logs/bias_fixed_' + cur_id + '.json', cur_id, dataset)
    
    
    #write_timeline(all_participants)
    #print metric_matrix
    #print metric_matrix['attribute_coverage']
#    plot_metric_heat_map(metric_matrix['data_point_coverage'], 'Data Point Coverage', 'Data Point', 'Time (Interactions)', directory + 'user_1506629987658/plots/', 'matrix_test5.png', 1)
    #plot_metric_heat_map(metric_matrix['data_point_distribution'], 'Data Point Distribution', 'Data Point', 'Time (Interactions)', directory + 'user_1507820577674/plots/', 'newdpcplot.png', 1)
    #plot_metric_heat_map(metric_matrix['attribute_coverage']['Weight (Pounds)'], 'Attribute Coverage (Weight)', 'Quantile', 'Time (Interactions)', directory + 'user_1506629987658/plots/', 'matrix_test5.png', 1)
    # emily: this isn't saving the fig?
    # emily remove this after testing
    
    #avg_bias_values(all_participants, 'avg')