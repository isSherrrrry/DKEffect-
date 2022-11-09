#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script is used to analyze classification data from 
the bias eval user study. It produces values against
which to compare the bias interaction metrics. 

Created on Mon Aug 21 00:13:06 2017

@author: emilywall
"""

import numpy as np
import matplotlib.pyplot as plt
import bias_util
import json
import csv
import copy
import sys
import os
import operator
from os import listdir
from os.path import isfile, isdir, join
from sklearn import svm
from sklearn import tree
from scipy import stats
from dateutil.parser import parse
import graphviz
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# train an SVM model using the given data and labels
# return the model weights and the classes
def get_svm_weights(data, labels): 
    data = np.array(data)
    labels = np.array(labels)
    clf = svm.SVC(kernel = 'linear') #(kernel = 'linear', C = .1)
    clf.fit(data, labels)
    results = clf.coef_
    weights = []
    for i in range(0, len(results)):
        weights.append(normalize_weights(list(results[i])))
    classes = clf.classes_
    
    return weights, classes
    
# normalize the set of weights as if all are positive, then add negative factors at the end
def normalize_weights(notNormedWeights):
    factors = []
    for i in range(len(notNormedWeights)):
        if notNormedWeights[i] < 0:
            notNormedWeights[i] *= -1
            factors.append(-1)
        else:
            factors.append(1)
    s = sum(notNormedWeights)
    pos_weights = [(r / s) for r in notNormedWeights]
    weights = [pos_weights[i] * factors[i] for i in range(len(factors))]
    return weights
    
# run SVM and write the results to the given directory
def write_svm_results(directory, file_name, log_file, to_plot, fig_num, verbose): 
    if (verbose):
        print 'Writing and Plotting SVM Data: ', file_name
        
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    all_data = dict()
    all_data['classifications'] = classification
    
    x_data = []
    y_data = []
    data = []   

    features = bias_util.get_bball_player(dataset, list(classification.keys())[0]).get_map().keys()
    features.remove('Name')
    features = sorted(features)
        
    for key in classification.keys():
        cur_player = bias_util.get_bball_player(dataset, key)
        cur_map = cur_player.get_map()
        cur_map['*Classification'] = classification[key]
        data.append(cur_map)
        
        cur_x = []
        for i in range(0, len(features)):
            cur_x.append(cur_map[features[i]])
        cur_x = [float(x) for x in cur_x]
        x_data.append(cur_x)
        y_data.append(bias_util.pos_to_num_map[classification[key]])
        
    svm_weights, svm_classes = get_svm_weights(x_data, y_data)
    weights_map = dict()
    i = 0
    for j in range(0, len(svm_classes)):
        for k in range(j + 1, len(svm_classes)):
            key = bias_util.num_to_pos_map[j] + ' - ' + bias_util.num_to_pos_map[k]
            value = svm_weights[i]
            weights_map[key] = value
            i += 1
    
    all_data['features'] = features
    all_data['weights'] = weights_map
    all_data['classifications'] = data

    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    f_out.write('{')
    f_out.write('"features":' + json.dumps(all_data['features']) + ',')
    f_out.write('"weights":' + json.dumps(all_data['weights']) + ',')
    f_out.write('"classifications":' + json.dumps(all_data['classifications']))
    f_out.write('}')
    f_out.close()
    
    if (to_plot == True):
        for key in weights_map.keys():
            plot_svm(features, weights_map[key], 'SVM Feature Weights: ' + key, 'Feature', 'Weight', directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png').replace('svm', 'svm_' + key), fig_num, verbose)
            fig_num += 1
            
    return svm_weights
    
# plot the weights from SVM as a bar chart
def plot_svm(features, weights, title, x_label, y_label, directory, file_name, fig_num, verbose):
    if (verbose): 
        print 'plotting', title
    
    plt.figure(num = fig_num, figsize = (5, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 1, 1)
    
    # create the plot
    y_pos = np.arange(len(features))
    plt.bar(y_pos, weights, align = 'center', alpha = 0.5)
    plt.xticks(y_pos, features, rotation = 'vertical')
    plt.tight_layout()
        
    # label, save, and clear
    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# compute the centroid data point given the list of data points
def compute_centroid(data):
    centroid = dict()
    attributes = data[0].keys()
    for attr in attributes:
        if (attr == 'Name' or attr == 'Name (Real)' or attr == 'Team' or attr == 'Position'): 
            continue
        else:
            avg = 0
            for i in range(0, len(data)):
                avg += float(data[i][attr])
            avg /= len(data)
            centroid[attr] = avg
        
    return centroid
    
# get the identification-confusion matrix from the final classification
# row (y) is user-defined label, col (x) is actual label
def get_id_confusion_matrix(logs, dataset):
    id_confusion = np.zeros((7, 7))
    all_data = dict()
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    
    for i in range(0, len(dataset)):
        cur_data = dataset[i].get_full_map()
        cur_id = cur_data['Name'].replace('Player ', '')
        actual_pos = cur_data['Position']
        if (cur_id in classification): 
            user_pos = classification[cur_id]
            id_confusion[bias_util.pos_to_num_map[user_pos], bias_util.pos_to_num_map[actual_pos]] += 1
            key = 'user:' + user_pos + ',actual:' + actual_pos
            if key in all_data: 
                all_data[key].append(cur_data)
            else: 
                all_data[key] = [cur_data]
                
    return id_confusion, bias_util.pos_to_num_map, all_data
    
# write the identification-confusion matrix to a json file
def write_id_confusion_matrix(directory, file_name, log_file, fig_num, verbose):
    if (verbose):
        print 'Writing and Plotting ID-Confusion Matrix Data: ', file_name
    
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    id_confusion, pos_to_num_map, all_data = get_id_confusion_matrix(logs, dataset)
  
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    summary = dict()
    summary['rows (y)'] = 'user'
    summary['cols (x)'] = 'actual'
    summary['position_indices'] = pos_to_num_map
    summary['centroids'] = dict()
    summary['centroids']['user_centroids'] = dict()
    summary['centroids']['actual_centroids'] = dict()

    # separate out user labels and actual labels
    user_labels = dict()
    actual_labels = dict()
    for key in all_data.keys(): 
        cur_user_label = key[5 : key.index(',')]
        cur_actual_label = key[key.index('actual') + 7 : ]
        cur_data_point = copy.deepcopy(all_data[key])

        if (cur_user_label in user_labels.keys()):
            user_labels[cur_user_label] += cur_data_point
        else: 
            user_labels[cur_user_label] = cur_data_point
        if (cur_actual_label in actual_labels.keys()):
            actual_labels[cur_actual_label] += cur_data_point
        else: 
            actual_labels[cur_actual_label] = cur_data_point

    # compute centroids
    for key in user_labels.keys():
        summary['centroids']['user_centroids'][key] = compute_centroid(user_labels[key])
        #if (verbose):
        #    print 'User Centroid', key, summary['centroids']['user_centroids'][key]
    for key in actual_labels.keys():
        summary['centroids']['actual_centroids'][key] = compute_centroid(actual_labels[key])
        #if (verbose):
        #    print 'Actual Centroid', key, summary['centroids']['actual_centroids'][key]

    # get total accuracy
    num_correct = 0
    total_classifications = 0
    for i in range(0, len(id_confusion)): 
        for j in range(0, len(id_confusion[i])):
            total_classifications += id_confusion[i][j]
            if (i == j): 
                num_correct += id_confusion[i][j]
    
    summary['total_accuracy'] = str(num_correct) + '/' + str(total_classifications)
    summary['matrix'] = id_confusion.tolist()
    f_out.write('{')
    f_out.write('"summary":' + json.dumps(summary) + ',')
    f_out.write('"all_data":' + json.dumps(all_data))
    f_out.write('}')
    f_out.close()
    
    # plot the matrix
    labels = [bias_util.num_to_pos_map[0], bias_util.num_to_pos_map[1], bias_util.num_to_pos_map[2], bias_util.num_to_pos_map[3], bias_util.num_to_pos_map[4]]
    plot_id_conf_matrix(id_confusion.tolist(), labels, directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png'), fig_num)
    
# plot the identification-confusion matrix
def plot_id_conf_matrix(matrix, labels, directory, file_name, fig_num):
    plt.figure(num = fig_num, figsize = (5, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 1, 1)
    matrix = np.array(matrix)
    matrix = matrix[0 : 5, 0 : 5]
    heatmap = plt.pcolor(matrix, cmap = 'Blues')
    
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%d' % matrix[y, x], horizontalalignment = 'center', verticalalignment = 'center')
    
    plt.colorbar(heatmap)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation = 'vertical')
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.xlabel('Actual Category')
    plt.ylabel('User Category')
    plt.title('Identification-Confusion Matrix')
    plt.tight_layout()
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# write the classification accuracy over time to a file
def write_classification_accuracy(directory, file_name, log_file, fig_num, verbose):
    print 'Writing and Plotting Accuracy Over Time: ', file_name
    
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    logs = item_logs
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(logs, dataset)
    
    total_labeled = 0
    total_correct = 0
    decision_points = np.arange(1, len(all_logs) + 1)
    accuracy = [-1] * len(all_logs)
    current_labels = dict()
    correct_labels = dict()
    
    for i in range(0, len(decisions_labels)):
        cur = decisions_labels[i]
        cur_id = cur[3]
        correct_labels[cur_id] = cur[2]
        
        if ((cur_id not in current_labels and cur[1] != 'Un-Assign') or (cur_id in current_labels and current_labels[cur_id] == 'Un-Assign' and cur[1] != 'Un-Assign')):
            total_labeled += 1
        elif (cur_id in current_labels and cur[1] == 'Un-Assign' and current_labels[cur_id] != 'Un-Assign'):
            total_labeled -= 1
            
        if (cur_id not in current_labels and cur[1] == correct_labels[cur_id]): 
            total_correct += 1
        elif (cur_id in current_labels and current_labels[cur_id] != correct_labels[cur_id] and cur[1] == correct_labels[cur_id]):
            total_correct += 1
            
        if (cur_id in current_labels and current_labels[cur_id] == correct_labels[cur_id] and cur[1] != correct_labels[cur_id]):
            total_correct -= 1
            
        if (total_labeled != 0):
            accuracy[cur[0]] = total_correct / float(total_labeled)
        else: 
            accuracy[cur[0]] = 0
        current_labels[cur_id] = cur[1]
    if (len(decisions_labels) < 1):
        first_decision = -1
    else: 
        first_decision = decisions_labels[0][0]
    accuracy = bias_util.remove_defaults(accuracy, first_decision)
    accuracy = bias_util.forward_fill(accuracy)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    f_out = open(directory + file_name, 'w+')
    f_out.write('[')
    for i in range(0, len(decisions_labels)):
        f_out.write('{')
        f_out.write('"interaction_number":"' + str(decisions_labels[i][0]) + '",')
        f_out.write('"data_point":"' + str(decisions_labels[i][3]) + '",')
        f_out.write('"actual_class":"' + str(decisions_labels[i][2]) + '",')
        f_out.write('"user_class":"' + str(decisions_labels[i][1]) + '",')
        f_out.write('"current_accuracy":"' + str(accuracy[decisions_labels[i][0]]) + '"')
        f_out.write('}')
        if (i != len(decisions_labels) - 1): 
            f_out.write(',')
    f_out.write(']')
    f_out.close()

    plot_classification_accuracy(decision_points, accuracy, 'Accuracy Over Time', 'Interactions', 'Accuracy', directory.replace('/logs/', '/plots/'), file_name.replace('.json', '.png'), decisions_labels, fig_num, verbose)
    
# plot the classification accuracy over time
def plot_classification_accuracy(x_values, y_values, title, x_label, y_label, directory, file_name, decisions, fig_num, verbose):    
    plt.figure(num = fig_num, figsize = (15, 5), dpi = 80, facecolor = 'w', edgecolor = 'k')
    sub_plot = plt.subplot(1, 1, 1)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    plt.plot(x_values, y_values, c = '#000000')
             
    for i in range(0, len(decisions)): 
        tup = decisions[i]
        sub_plot.axvline(x = tup[0], c = bias_util.color_map[tup[1]], linewidth = 2, zorder = 0, clip_on = False)
    
    # label, save, and clear
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + file_name)
    plt.clf()
    
# get the similarity of two metric sequences by using dynamic time warping 
# input sequences are just lists of numbers; function needs to zip them with list indices
def get_dtw_similarity(y1, y2):
    x1 = np.arange(1, len(y1) + 1)
    x2 = np.arange(1, len(y2) + 1)
    
    seq1 = np.array(zip(x1, y1))
    seq2 = np.array(zip(x2, y2))
    
    distance, path = fastdtw(seq1, seq2, dist = euclidean)
    print '  ' + str(seq1)
    print '  ' + str(seq2)
    print path #emily remove
    return distance, path
    
# perform dynamic time warping between users with the metrics
def dynamic_time_warping(verbose):
    if (verbose):
        print '** Creating DTW Map'
    all_users = [f.replace('user_', '').replace('.json', '') for f in listdir(bias_util.directory) if ('user' in f and isdir(join(bias_util.directory, f)))]
    all_metrics = ['data_point_coverage', 'data_point_distribution']
    for i in range(0, len(bias_util.attrs)):
        all_metrics.append('attribute_coverage_' + bias_util.attrs[i])
        all_metrics.append('attribute_distribution_' + bias_util.attrs[i])
        all_metrics.append('attribute_weight_coverage_' + bias_util.attrs[i])
        all_metrics.append('attribute_weight_distribution_' + bias_util.attrs[i])

    dtw_metric_map = dict()
    
    # read each user's bias metrics for each window method and each metric
    # compare to each other via DTW
    for i in range(0, len(bias_util.window_methods)):
        window_method = bias_util.window_methods[i]
        dtw_metric_map[window_method] = dict()
      
        if (verbose):
            print '    Window Method:', window_method
            
        for j in range(0, len(all_users)):
            cur_user = all_users[j]
            cur_file_name = bias_util.directory + 'user_' + cur_user + '/logs/bias_' + window_method + '_' + cur_user + '.json'
            cur_file = json.loads(open(cur_file_name).read())
            
            dtw_metric_map[window_method][cur_user] = dict()
            for k in range(0, len(all_metrics)):
                cur_metric = all_metrics[k]
                dtw_metric_map[window_method][cur_user][cur_metric] = []
            
            for l in range(0, len(cur_file)):
                for k in range(0, len(all_metrics)):
                    cur_metric = all_metrics[k]
                    metric_vals = dtw_metric_map[window_method][cur_user][cur_metric] 
                    if ('data_point' in cur_metric and cur_metric in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics'][cur_metric]['metric_level'])
                    elif ('attribute_coverage' in cur_metric and 'attribute_coverage' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_coverage']['info']['attribute_vector'][cur_metric.replace('attribute_coverage_', '')]['metric_level'])
                    elif ('attribute_distribution' in cur_metric and 'attribute_distribution' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_distribution']['info']['attribute_vector'][cur_metric.replace('attribute_distribution_', '')]['metric_level'])
                    elif ('attribute_weight_coverage' in cur_metric and 'attribute_weight_coverage' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_weight_coverage']['info']['attribute_vector'][cur_metric.replace('attribute_weight_coverage_', '')]['metric_level'])
                    elif ('attribute_weight_distribution' in cur_metric and 'attribute_weight_distribution' in cur_file[l]['bias_metrics']):
                        metric_vals.append(cur_file[l]['bias_metrics']['attribute_weight_distribution']['info']['attribute_vector'][cur_metric.replace('attribute_weight_distribution_', '')]['metric_level'])
                    dtw_metric_map[window_method][cur_user][cur_metric] = metric_vals
                    
    return dtw_metric_map, all_users, all_metrics

# run dynamic time warping and write the results to a file
def write_dtw(out_dir, verbose):
    dtw_metric_map, all_users, all_metrics = dynamic_time_warping(verbose)
    #f_out = open(out_dir + 'dtw_info.json', 'w+')
    #f_out.write('{')
    #f_out.write(json.dumps(dtw_metric_map))
    #f_out.write('}')
    #f_out.close()
    
    if (verbose):
        print '** DTW Map Done'
        print '** Writing Files'

    for i in range(0, len(bias_util.window_methods)):
        window_method = bias_util.window_methods[i]
        f_out = open(out_dir + 'post_hoc_' + window_method + '.csv', 'w+')
        if (verbose):
            print '    Writing File:', out_dir + 'post_hoc_' + window_method + '.csv'
            
        header = 'user1,user2,'
        for j in range(0, len(all_metrics)):
            header += all_metrics[j]
            if (j != len(all_metrics) - 1):
                header += ','
        f_out.write(header + '\n')
        
        for a in range(0, len(all_users)): 
            user1 = all_users[a]
            for b in range(a + 1, len(all_users)): 
                user2 = all_users[b]
                line = user1 + ',' + user2 + ','
                for j in range(0, len(all_metrics)):
                    cur_metric = all_metrics[j]
                    y1 = dtw_metric_map[window_method][user1][cur_metric]
                    y2 = dtw_metric_map[window_method][user2][cur_metric]
                    dist, path = get_dtw_similarity(y1, y2)
                    if (verbose):
                        print '        DTW ', cur_metric, ' Dist:', user1, '--', user2, '=', dist
                    line += str(dist)
                    if (j != len(all_metrics) - 1):
                        line += ','
                f_out.write(line + '\n')
                    
        f_out.close()
        
# predict the order of position classifications
# if error: make sure clicks aren't filtered!
def classification_prediction(directories, log_files, user_ids, out_dir, verbose):
    class_counts = dict()
    if (not isinstance(directories, (tuple, list))):
        X, Y, class_counts = one_classification_prediction(directories, log_files, user_ids, class_counts, verbose)
        out_file = 'graphviz_test.dot'# Emily temp remove_' + user_ids + '.dot'
        if (verbose): 
            print 'Classification Prediction: ', user_ids
            print 'counts', class_counts
    else: 
        X = []
        Y = []
        out_file = 'graphviz_test.dot' # emily temp
        for i in range(0, len(directories)):
            if (verbose): 
                print 'Classification Prediction: ', user_ids[i]
            x, y, class_counts = one_classification_prediction(directories[i], log_files[i], user_ids[i], class_counts, verbose)
            X = X + x
            Y = Y + y
            print directories[i]
            
    class_names = bias_util.pos_names[:]
    class_names_trimmed = []
    for key in class_names:
        if (key in class_counts):
            class_names_trimmed.append(key)
            
    print 'classes (', len(class_names_trimmed), '): ', class_names_trimmed
     
    feature_names = ['prev_class']
    out_dir = '/Users/emilywall/git/bias_eval/py/'# emily temp
    decision_tree(X, Y, out_dir, out_file, feature_names, class_names_trimmed, verbose)
            
# predict the order of classification given the current classification using a decision tree for a single user
def one_classification_prediction(directory, log_file, user_id, class_counts, verbose):
    data_directory = bias_util.data_directory
    data_file_name = bias_util.data_file_name
    
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    dataset, attr_map = bias_util.read_data(data_directory, data_file_name)
    classification, decisions_labels, decisions_cat, decisions_help = bias_util.get_classifications_and_decisions(all_logs, dataset)
    
    X = []
    Y = []
    
    # decisions_labels of the form (index, user classification, actual classification, player id)
    prev = -1
    cur = -1
    for i in range(0, len(decisions_labels)):     
        prev = cur
        cur = decisions_labels[i][1]
        
        if (not cur in class_counts):
            class_counts[cur] = 1
        else: 
            class_counts[cur] += 1
            
        if (prev != -1 and cur != -1):# and prev != cur): # create a training instance
            # comment prev != cur condition to allow repetitions
            X.append([bias_util.pos_to_num_map[prev]])
            Y.append(bias_util.pos_to_num_map[cur])
            
    return X, Y, class_counts
        
# compile the training data from potentially multiple users to predict interactions
def interaction_prediction(directories, log_files, user_ids, out_dir, verbose):
    int_counts = dict()
    if (not isinstance(directories, (tuple, list))):
        if (verbose): 
            print 'Interaction Prediction: ', user_ids
        X, Y, int_counts = one_interaction_prediction(directories, log_files, user_ids, int_counts, verbose)
        out_file = 'graphviz_test.dot'# Emily temp remove_' + user_ids + '.dot'
    else: 
        X = []
        Y = []
        out_file = 'graphviz_test.dot' # emily temp
        for i in range(0, len(directories)):
            if (verbose): 
                print 'Interaction Prediction: ', user_ids[i]
            x, y, int_counts = one_interaction_prediction(directories[i], log_files[i], user_ids[i], int_counts, verbose)
            X = X + x
            Y = Y + y
            print directories[i]
        
    class_names = bias_util.interaction_names[:]
    class_names_trimmed = []
    for key in class_names:
        if (key in int_counts):
            class_names_trimmed.append(key)
    
    feature_names = ['prev_interaction']
    out_dir = '/Users/emilywall/git/bias_eval/py/'# emily temp
    decision_tree(X, Y, out_dir, out_file, feature_names, class_names_trimmed, verbose)
    
# predict the next interaction mode given the current interaction mode using a decision tree for a single user
def one_interaction_prediction(directory, log_file, user_id, int_counts, verbose): 
    all_logs, attr_logs, item_logs, help_logs, cat_logs  = bias_util.recreate_logs(directory, log_file)
    X = []
    Y = []
    prev = -1
    cur = -1
    int_to_skip = ['set_attribute_weight_vector_init']
    
    for i in range(0, len(all_logs)):
        prev = cur
        cur = all_logs[i]['customLogInfo']['eventType']
        
        if (cur in int_to_skip or prev in int_to_skip):
            continue # skip this interaction
            
        if (not cur in int_counts):
            int_counts[cur] = 1
        else: 
            int_counts[cur] += 1
                
        if (prev != -1 and cur != -1):# and prev != cur): # create a training instance
            # comment prev != cur condition to allow repetitions
            X.append([bias_util.int_to_num_map[prev]])
            Y.append(bias_util.int_to_num_map[cur])
            
    return X, Y, int_counts
            
    
# create a decision tree with the given training data
def decision_tree(X, Y, directory, file_name, feature_names, class_names, verbose):
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X, Y)
    if (feature_names != -1 and class_names != -1):
        dot_data = tree.export_graphviz(classifier, out_file = directory + file_name, feature_names = feature_names, class_names = class_names) #emily: file name should end with user's id + .dot 
    else: 
        dot_data = tree.export_graphviz(classifier, out_file = directory + file_name) #emily: file name should end with user's id + .dot 
    graph = graphviz.Source(dot_data) 
    graph.render('graphviz_test', view = True)
    if (verbose):
        print '  Done: Decision Tree ', file_name
  
      
# get the sequence of axis changes per participant
# keep track of (1) the number of times each axis is selected,
# (2) the added weight of each attribute, (3) the added weight
# of each attribute scaled by time that it was selected
def axis_changes(ids):
    all_users_axis_changes = dict()
    computations = ['num_selections', 'time_raw', 'time_scaled']
    conditions = ['cond_size', 'cond_role', 'size-role']
    
    all_users_summary = dict()
    for i in range(0, len(computations)):
        all_users_summary[computations[i]] = dict()
        
        for j in range(0, len(conditions)):
            all_users_summary[computations[i]][conditions[j]] = dict()
            all_users_summary[computations[i]][conditions[j]]['summary'] = dict()
            all_users_summary[computations[i]][conditions[j]]['summary']['custom'] = 0
            
            for k in range(0, len(bias_util.attrs)):
                all_users_summary[computations[i]][conditions[j]]['summary'][bias_util.attrs[k]] = 0
        
        for j in range(0, len(bias_util.cond_size)):
            all_users_summary[computations[i]]['cond_size'][str(bias_util.cond_size[j])] = dict()
            all_users_summary[computations[i]]['cond_size'][str(bias_util.cond_size[j])]['custom'] = 0            
            
            for k in range(0, len(bias_util.attrs)):
                all_users_summary[computations[i]]['cond_size'][str(bias_util.cond_size[j])][bias_util.attrs[k]] = 0
        
        for j in range(0, len(bias_util.cond_role)):
            all_users_summary[computations[i]]['cond_role'][str(bias_util.cond_role[j])] = dict()
            all_users_summary[computations[i]]['cond_role'][str(bias_util.cond_role[j])]['custom'] = 0
            
            for k in range(0, len(bias_util.attrs)):
                all_users_summary[computations[i]]['cond_role'][str(bias_util.cond_role[j])][bias_util.attrs[k]] = 0
    
    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        
        all_axis_changes = []
        axis_selections = []
        cur_file_name = bias_util.directory + 'user_' + cur_user + '/logs/interactions_' + cur_user + '.json'
        cur_file = json.loads(open(cur_file_name).read())
        cur_user_total_time = parse(cur_file[-1]['eventTimeStamp']) - parse(cur_file[0]['eventTimeStamp'])
        cur_user_total_time = cur_user_total_time.total_seconds()
        
        if (ids[i] in bias_util.cond_size):
            all_users_summary['num_selections']['cond_size'][cur_user]['*total_time*'] = cur_user_total_time
            all_users_summary['time_raw']['cond_size'][cur_user]['*total_time*'] = cur_user_total_time
            all_users_summary['time_scaled']['cond_size'][cur_user]['*total_time*'] = cur_user_total_time
        
        else:
            all_users_summary['num_selections']['cond_role'][cur_user]['*total_time*'] = cur_user_total_time
            all_users_summary['time_raw']['cond_role'][cur_user]['*total_time*'] = cur_user_total_time
            all_users_summary['time_scaled']['cond_role'][cur_user]['*total_time*'] = cur_user_total_time
        
        # go through each interaction in time
        for j in range(0, len(cur_file)):
            if ('AttributeWeightChange' in cur_file[j]['eventName']):
                if (cur_file[j]['customLogInfo']['eventType'] == 'set_attribute_weight_vector_select'):
                    for k in range(0, len(bias_util.attrs)):
                        if (cur_file[j]['newWeight'][bias_util.attrs[k]] == 1):
                            all_axis_changes.append({'attr': bias_util.attrs[k], 'time': cur_file[j]['eventTimeStamp'], 'type': 'select'})
                            axis_selections.append({'attr': bias_util.attrs[k], 'time': cur_file[j]['eventTimeStamp']})
                else: 
                    all_axis_changes.append({'attr': 'custom', 'weight': cur_file[j]['newWeight'], 'time': cur_file[j]['eventTimeStamp'], 'type': 'custom'})
                            
        # now tally up axis changes from the arrays
        for j in range(0, len(all_axis_changes)):
            if (ids[i] in bias_util.cond_size):
                all_users_summary['num_selections']['cond_size']['summary'][all_axis_changes[j - 1]['attr']] += 1
                all_users_summary['num_selections']['cond_size'][cur_user][all_axis_changes[j - 1]['attr']] += 1
            else: 
                all_users_summary['num_selections']['cond_role']['summary'][all_axis_changes[j - 1]['attr']] += 1
                all_users_summary['num_selections']['cond_role'][cur_user][all_axis_changes[j - 1]['attr']] += 1

            if (j > 0):
                time_diff = parse(all_axis_changes[j]['time']) - parse(all_axis_changes[j - 1]['time'])
                time_diff = time_diff.total_seconds() 
                time_diff_scaled = time_diff / cur_user_total_time
                
                if (ids[i] in bias_util.cond_size):
                    all_users_summary['time_raw']['cond_size']['summary'][all_axis_changes[j - 1]['attr']] += time_diff
                    all_users_summary['time_raw']['cond_size'][cur_user][all_axis_changes[j - 1]['attr']] += time_diff
                    
                    all_users_summary['time_scaled']['cond_size']['summary'][all_axis_changes[j - 1]['attr']] += time_diff_scaled
                    all_users_summary['time_scaled']['cond_size'][cur_user][all_axis_changes[j - 1]['attr']] += time_diff_scaled
                
                else: 
                    all_users_summary['time_raw']['cond_role']['summary'][all_axis_changes[j - 1]['attr']] += time_diff
                    all_users_summary['time_raw']['cond_role'][cur_user][all_axis_changes[j - 1]['attr']] += time_diff
                    all_users_summary['time_scaled']['cond_role']['summary'][all_axis_changes[j - 1]['attr']] += time_diff_scaled
                    all_users_summary['time_scaled']['cond_role'][cur_user][all_axis_changes[j - 1]['attr']] += time_diff_scaled
                    
        # account for time spent on final axis configuration
        final_time = cur_file[-1]['eventTimeStamp']
        time_diff = parse(final_time) - parse(all_axis_changes[-1]['time'])
        time_diff = time_diff.total_seconds() 
        time_diff_scaled = time_diff / cur_user_total_time
        
        if (ids[i] in bias_util.cond_size):
            all_users_summary['time_raw']['cond_size']['summary'][all_axis_changes[-1]['attr']] += time_diff
            all_users_summary['time_raw']['cond_size'][cur_user][all_axis_changes[-1]['attr']] += time_diff
            all_users_summary['time_scaled']['cond_size']['summary'][all_axis_changes[-1]['attr']] += time_diff_scaled
            all_users_summary['time_scaled']['cond_size'][cur_user][all_axis_changes[-1]['attr']] += time_diff_scaled
        
        else: 
            all_users_summary['time_raw']['cond_role']['summary'][all_axis_changes[-1]['attr']] += time_diff
            all_users_summary['time_raw']['cond_role'][cur_user][all_axis_changes[-1]['attr']] += time_diff
            all_users_summary['time_scaled']['cond_role']['summary'][all_axis_changes[-1]['attr']] += time_diff_scaled
            all_users_summary['time_scaled']['cond_role'][cur_user][all_axis_changes[-1]['attr']] += time_diff_scaled
            
        all_users_axis_changes[cur_user] = axis_selections
        
    # average per participant + get the difference between size and variable condition
    for i in range(0, len(bias_util.attrs)):
        # divide all by number of participants to get average per person
        all_users_summary['num_selections']['cond_size']['summary'][bias_util.attrs[i]] /= 5.0
        all_users_summary['num_selections']['cond_role']['summary'][bias_util.attrs[i]] /= 5.0
        all_users_summary['time_raw']['cond_size']['summary'][bias_util.attrs[i]] /= 5.0
        all_users_summary['time_raw']['cond_role']['summary'][bias_util.attrs[i]] /= 5.0
        all_users_summary['time_scaled']['cond_size']['summary'][bias_util.attrs[i]] /= 5.0
        all_users_summary['time_scaled']['cond_role']['summary'][bias_util.attrs[i]] /= 5.0
        
        # get the difference between the two conditions
        all_users_summary['num_selections']['size-role']['summary'][bias_util.attrs[i]] = all_users_summary['num_selections']['cond_size']['summary'][bias_util.attrs[i]] - all_users_summary['num_selections']['cond_role']['summary'][bias_util.attrs[i]]
        all_users_summary['time_raw']['size-role']['summary'][bias_util.attrs[i]] = all_users_summary['time_raw']['cond_size']['summary'][bias_util.attrs[i]] - all_users_summary['time_raw']['cond_role']['summary'][bias_util.attrs[i]]
        all_users_summary['time_scaled']['size-role']['summary'][bias_util.attrs[i]] = all_users_summary['time_scaled']['cond_size']['summary'][bias_util.attrs[i]] - all_users_summary['time_scaled']['cond_role']['summary'][bias_util.attrs[i]]
    
    # for custom axes: divide all by number of participants to get average per person
    all_users_summary['num_selections']['cond_size']['summary']['custom'] /= 5.0
    all_users_summary['num_selections']['cond_role']['summary']['custom'] /= 5.0
    all_users_summary['time_raw']['cond_size']['summary']['custom'] /= 5.0
    all_users_summary['time_raw']['cond_role']['summary']['custom'] /= 5.0
    all_users_summary['time_scaled']['cond_size']['summary']['custom'] /= 5.0
    all_users_summary['time_scaled']['cond_role']['summary']['custom'] /= 5.0
    
    # get the difference between the two conditions
    all_users_summary['num_selections']['size-role']['summary']['custom'] = all_users_summary['num_selections']['cond_size']['summary']['custom'] - all_users_summary['num_selections']['cond_role']['summary']['custom']
    all_users_summary['time_raw']['size-role']['summary']['custom'] = all_users_summary['time_raw']['cond_size']['summary']['custom'] - all_users_summary['time_raw']['cond_role']['summary']['custom']
    all_users_summary['time_scaled']['size-role']['summary']['custom'] = all_users_summary['time_scaled']['cond_size']['summary']['custom'] - all_users_summary['time_scaled']['cond_role']['summary']['custom']
    
    f_out = open(bias_util.directory + 'analysis/axis_selections.json', 'w+')
    f_out.write(json.dumps(all_users_axis_changes))
    f_out.close()
    
    f_out = open(bias_util.directory + 'analysis/axis_change_summary.json', 'w+')
    f_out.write(json.dumps(all_users_summary))
    f_out.close()
    
    print '** Axis Change File Written'
    return all_users_axis_changes

# analyze average and standard deviation for bias values in each quantile
# when labeling C v. non-C, PG v. non-PG, etc.
def quantile_analysis(ids):
    all_users = dict()
    all_users_summary = dict()
    positions = ['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center']
    conditions = ['cond_size', 'cond_role']
    metrics = ['attribute_coverage', 'attribute_distribution']
    
    for i in range(0, len(positions)):
        all_users[positions[i]] = dict()
        all_users_summary[positions[i]] = dict()
        for j in range(0, len(conditions)):
            cur_cond = conditions[j]
            all_users[positions[i]][conditions[j]] = dict()
            all_users_summary[positions[i]][conditions[j]] = dict()
            all_users_summary[positions[i]][conditions[j]]['summary'] = dict()
            for k in range(0, len(metrics)):
                all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]] = dict()
                for l in range(0, len(bias_util.framed_attrs)):
                    all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]] = dict()
                    for m in range(0, 4):
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)] = dict()
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)]['mean'] = 0
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)]['stdev'] = 0
            user_list = bias_util.cond_size
            if (cur_cond != 'cond_size'):
                user_list = bias_util.cond_role
            for k in range(0, len(user_list)):
                cur_user = str(user_list[k])
                all_users[positions[i]][cur_cond][cur_user] = dict()
                for l in range(0, len(metrics)):
                    all_users[positions[i]][cur_cond][cur_user][metrics[l]] = dict()
                    for m in range(0, len(bias_util.framed_attrs)):
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]] = dict()
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['individual'] = dict()
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['mean'] = dict()
                        for n in range(0, 4):
                            all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['individual']['Q' + str(n + 1)] = []
                            all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['mean']['Q' + str(n + 1)] = 0

    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        print '*', cur_user
        cur_cond = 'cond_size'
        if (ids[i] not in bias_util.cond_size):
            cur_cond = 'cond_role'
        
        for j in range(0, len(bias_util.framed_attrs)):
            cur_attr = bias_util.framed_attrs[j]

            for k in range(0, len(metrics)):
                cur_metric = metrics[k]
                
                cur_file_name = bias_util.directory + 'user_' + cur_user + '/logs/metric_matrices/' + cur_metric + '_' + cur_attr + '.csv'
                cur_file = open(cur_file_name, 'rb')
                cur_file_reader = csv.reader(cur_file)
                                
                # figure out what the first decision was
                first_dec = 'none'
                first_row = True
                for row in cur_file_reader:
                    if (first_row == True):
                        first_row = False
                    else:
                        if (row[3] != 'none' and row[3] != 'Un-Assign'):
                            first_dec = row[3]
                            break
                cur_file.close()
                
                cur_file = open(cur_file_name, 'rb')
                cur_file_reader = csv.reader(cur_file)
                cur_class = first_dec
                first_row = True
                for row in cur_file_reader: 
                    if (first_row == True):
                        first_row = False
                    else:
                        cur_quant = 'Q' + row[0][0]
                        if (row[3] != 'none' and row[3] != 'Un-Assign'):
                            cur_class = row[3]
                        all_users[cur_class][cur_cond][cur_user][cur_metric][cur_attr]['individual'][cur_quant].append(float(row[2]))

    # average per participant + average within the size and variable condition
    for i in range(0, len(conditions)):
        cur_condition = conditions[i]
        user_list = bias_util.cond_size
        if (cur_condition != 'cond_size'):
            user_list = bias_util.cond_role
            
        for j in range(0, len(positions)):
            cur_position = positions[j]
            for k in range(0, len(metrics)):
                cur_metric = metrics[k]
                for l in range(0, len(bias_util.framed_attrs)):
                    cur_attr = bias_util.framed_attrs[l]
                    for m in range(0, 4):
                        cur_quant = 'Q' + str(m + 1)
                        metric_vals = []
                        for n in range(0, len(user_list)):
                            cur_user = str(user_list[n])
                            all_users[cur_position][cur_condition][cur_user][cur_metric][cur_attr]['mean'][cur_quant] = np.mean(all_users[cur_position][cur_condition][cur_user][cur_metric][cur_attr]['individual'][cur_quant])
                            metric_vals.append(all_users[cur_position][cur_condition][cur_user][cur_metric][cur_attr]['mean'][cur_quant])
                        all_users_summary[cur_position][cur_condition]['summary'][cur_metric][cur_attr][cur_quant]['mean'] = np.mean(metric_vals)
                        all_users_summary[cur_position][cur_condition]['summary'][cur_metric][cur_attr][cur_quant]['stdev'] = np.std(metric_vals)
                        
    f_out = open(bias_util.directory + 'analysis/json/quantile_analysis_all.json', 'w+')
    f_out.write(json.dumps(all_users))
    f_out.close()
    
    f_out = open(bias_util.directory + 'analysis/json/quantile_analysis_summary.json', 'w+')
    f_out.write(json.dumps(all_users_summary))
    f_out.close()
    
    # now separate out just the csv info we need for comparison by condition
    f_out = open(bias_util.directory + 'analysis/csv/quantile_analysis_conditions.csv', 'w+')
    f_out.write('metric,size,,,,role,,,\n')
    f_out.write(',Q1,Q2,Q3,Q4,Q1,Q2,Q3,Q4\n')
    metrics = ['attribute_coverage', 'attribute_distribution']
    framed_attrs_role = ['Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds']
    framed_attrs_role_positions = ['Shooting Guard', 'Shooting Guard', 'Point Guard', 'Center', 'Power Forward', 'Small Forward', 'Power Forward']
    framed_attrs_size = ['Height (Inches)', 'Weight (Pounds)']
    all_positions = ['Center', 'Power Forward', 'Small Forward', 'Shooting Guard', 'Point Guard']
    for i in range(0, len(metrics)):
        for j in range(0, len(framed_attrs_role)):
            line = metrics[i] + ' - ' + framed_attrs_role[j] + '(' + framed_attrs_role_positions[j] + ')'
            size_quants = []
            role_quants = []
            for k in range(0, 4):
                size_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_size']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                role_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
            for k in range(0, 4):
                line += ',' + str(size_quants[k])
            for k in range(0, 4):
                line += ',' + str(role_quants[k])
            f_out.write(line + '\n')
            
        for j in range(0, len(framed_attrs_size)):
            for k in range(0, len(all_positions)):
                line = metrics[i] + ' - ' + framed_attrs_size[j] + '(' + all_positions[k] + ')'
                size_quants = []
                role_quants = []
                for l in range(0, 4):
                    size_quants.append(all_users_summary[all_positions[k]]['cond_size']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    role_quants.append(all_users_summary[all_positions[k]]['cond_role']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                for l in range(0, 4):
                    line += ',' + str(size_quants[l])
                for l in range(0, 4):
                    line += ',' + str(role_quants[l])
                f_out.write(line + '\n')
    f_out.close()
    
    # now separate out just the csv info we need for comparison by position
    f_out = open(bias_util.directory + 'analysis/csv/quantile_analysis_positions.csv', 'w+')
    f_out.write('metric,position,,,,non-position,,,\n')
    f_out.write(',Q1,Q2,Q3,Q4,Q1,Q2,Q3,Q4\n')
    metrics = ['attribute_coverage', 'attribute_distribution']
    framed_attrs_role = ['Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds']
    framed_attrs_role_positions = ['Shooting Guard', 'Shooting Guard', 'Point Guard', 'Center', 'Power Forward', 'Small Forward', 'Power Forward']
    framed_attrs_size = ['Height (Inches)', 'Weight (Pounds)']
    all_positions = ['Center', 'Power Forward', 'Small Forward', 'Shooting Guard', 'Point Guard']
    for i in range(0, len(metrics)):
        for j in range(0, len(framed_attrs_role)):
            line = metrics[i] + ' - ' + framed_attrs_role[j] + '(' + framed_attrs_role_positions[j] + ')'
            pos_quants = []
            non_pos_quants = []
            for k in range(0, 4):
                pos_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                cur_non_pos_quant = []
                for l in range(0, len(all_positions)):
                    if (all_positions[l] != framed_attrs_role_positions[j]):
                        cur_non_pos_quant.append(all_users_summary[all_positions[l]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                non_pos_quants.append(np.mean(cur_non_pos_quant))
            for k in range(0, 4):
                line += ',' + str(pos_quants[k])
            for k in range(0, 4):
                line += ',' + str(non_pos_quants[k])
            f_out.write(line + '\n')
            
        for j in range(0, len(framed_attrs_size)):
            for k in range(0, len(all_positions)):
                line = metrics[i] + ' - ' + framed_attrs_size[j] + '(' + all_positions[k] + ')'
                pos_quants = []
                non_pos_quants = []
                for l in range(0, 4):
                    pos_quants.append(all_users_summary[all_positions[k]]['cond_size']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    cur_non_pos_quant = []
                    for m in range(0, len(all_positions)):
                        if (all_positions[m] != all_positions[k]):
                            cur_non_pos_quant.append(all_users_summary[all_positions[m]]['cond_role']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    non_pos_quants.append(np.mean(cur_non_pos_quant))
                for l in range(0, 4):
                    line += ',' + str(pos_quants[l])
                for l in range(0, 4):
                    line += ',' + str(non_pos_quants[l])
                f_out.write(line + '\n')
    f_out.close()
    
    print '** Quantile Analysis Files Written'
    return all_users_summary

# analyze average and standard deviation for bias values in each quantile
# when labeling C v. non-C, PG v. non-PG, etc. 
# maintaining information per participant
def quantile_analysis_per_participant(ids):
    all_users = dict()
    all_users_summary = dict()
    positions = ['Point Guard', 'Shooting Guard', 'Small Forward', 'Power Forward', 'Center']
    conditions = ['cond_size', 'cond_role']
    metrics = ['attribute_coverage', 'attribute_distribution']
    
    for i in range(0, len(positions)):
        all_users[positions[i]] = dict()
        all_users_summary[positions[i]] = dict()
        for j in range(0, len(conditions)):
            cur_cond = conditions[j]
            all_users[positions[i]][conditions[j]] = dict()
            all_users_summary[positions[i]][conditions[j]] = dict()
            all_users_summary[positions[i]][conditions[j]]['summary'] = dict()
            for k in range(0, len(metrics)):
                all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]] = dict()
                for l in range(0, len(bias_util.framed_attrs)):
                    all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]] = dict()
                    for m in range(0, 4):
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)] = dict()
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)]['mean'] = 0
                        all_users_summary[positions[i]][conditions[j]]['summary'][metrics[k]][bias_util.framed_attrs[l]]['Q' + str(m + 1)]['stdev'] = 0
            user_list = bias_util.cond_size
            if (cur_cond != 'cond_size'):
                user_list = bias_util.cond_role
            for k in range(0, len(user_list)):
                cur_user = str(user_list[k])
                all_users[positions[i]][cur_cond][cur_user] = dict()
                for l in range(0, len(metrics)):
                    all_users[positions[i]][cur_cond][cur_user][metrics[l]] = dict()
                    for m in range(0, len(bias_util.framed_attrs)):
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]] = dict()
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['individual'] = dict()
                        all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['mean'] = dict()
                        for n in range(0, 4):
                            all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['individual']['Q' + str(n + 1)] = []
                            all_users[positions[i]][cur_cond][cur_user][metrics[l]][bias_util.framed_attrs[m]]['mean']['Q' + str(n + 1)] = 0

    for i in range(0, len(ids)):
        cur_user = str(ids[i])
        print '*', cur_user
        cur_cond = 'cond_size'
        if (ids[i] not in bias_util.cond_size):
            cur_cond = 'cond_role'
        
        for j in range(0, len(bias_util.framed_attrs)):
            cur_attr = bias_util.framed_attrs[j]

            for k in range(0, len(metrics)):
                cur_metric = metrics[k]
                
                cur_file_name = bias_util.directory + 'user_' + cur_user + '/logs/metric_matrices/' + cur_metric + '_' + cur_attr + '.csv'
                cur_file = open(cur_file_name, 'rb')
                cur_file_reader = csv.reader(cur_file)
                                
                # figure out what the first decision was
                first_dec = 'none'
                first_row = True
                for row in cur_file_reader:
                    if (first_row == True):
                        first_row = False
                    else:
                        if (row[3] != 'none' and row[3] != 'Un-Assign'):
                            first_dec = row[3]
                            break
                cur_file.close()
                
                cur_file = open(cur_file_name, 'rb')
                cur_file_reader = csv.reader(cur_file)
                cur_class = first_dec
                first_row = True
                for row in cur_file_reader: 
                    if (first_row == True):
                        first_row = False
                    else:
                        cur_quant = 'Q' + row[0][0]
                        if (row[3] != 'none' and row[3] != 'Un-Assign'):
                            cur_class = row[3]
                        all_users[cur_class][cur_cond][cur_user][cur_metric][cur_attr]['individual'][cur_quant].append(float(row[2]))

    f_out = open(bias_util.directory + 'analysis/csv/quantile_analysis_conditions_per_participant.csv', 'w+')
    #f_out.write('user_id,framing_condition,metric,attribute,position,framing_position,quantile,mean,hmean,stdev\n')
    f_out.write('user_id,framing_condition,metric,attribute,position,framing_position,quantile,mean,stdev\n')
    frame_map = dict()
    frame_map['cond_role'] = { 'Center': 'Avg. Blocks', 'Power Forward': 'Avg. Offensive Rebounds / Avg. Total Rebounds', 'Small Forward': 'Avg. Steals', 'Shooting Guard': 'Avg. 3-Pointers Att. / Avg. 3-Pointers Made', 'Point Guard': 'Avg. Assists' }
    frame_map['cond_size'] = { 'Center': 'Height (Inches) / Weight (Pounds)', 'Power Forward': 'Height (Inches) / Weight (Pounds)', 'Small Forward': 'Height (Inches) / Weight (Pounds)', 'Shooting Guard': 'Height (Inches) / Weight (Pounds)', 'Point Guard': 'Height (Inches) / Weight (Pounds)' }

    # average per participant
    for i in range(0, len(conditions)):
        cur_condition = conditions[i]
        user_list = bias_util.cond_size
        if (cur_condition != 'cond_size'):
            user_list = bias_util.cond_role
            
        for j in range(0, len(positions)):
            cur_position = positions[j]
            cur_framing_position = str(frame_map[cur_condition][cur_position])
            print cur_framing_position
            for k in range(0, len(metrics)):
                cur_metric = metrics[k]
                for l in range(0, len(bias_util.framed_attrs)):
                    cur_attr = bias_util.framed_attrs[l]
                    for m in range(0, 4):
                        cur_quant = 'Q' + str(m + 1)
                        for n in range(0, len(user_list)):
                            cur_user = str(user_list[n])
                            cur_vals = all_users[cur_position][cur_condition][cur_user][cur_metric][cur_attr]['individual'][cur_quant]
                            #print '**', cur_vals
                            cur_mean = np.mean(cur_vals)
                            #cur_hmean = stats.hmean(cur_vals)
                            cur_stdev = np.std(cur_vals)
                            #f_out.write(cur_user + ',' + cur_condition + ',' + cur_metric + ',' + cur_attr + ',' + cur_position + ',' + cur_framing_position + ',' + cur_quant + ',' + str(cur_mean) + ',' + str(cur_hmean) + ',' + str(cur_stdev) + '\n')
                            f_out.write(cur_user + ',' + cur_condition + ',' + cur_metric + ',' + cur_attr + ',' + cur_position + ',' + cur_framing_position + ',' + cur_quant + ',' + str(cur_mean) + ',' + str(cur_stdev) + '\n')
    f_out.close()
    
    # now separate out just the csv info we need for comparison by condition
    '''
    f_out = open(bias_util.directory + 'analysis/csv/quantile_analysis_conditions.csv', 'w+')
    f_out.write('metric,size,,,,role,,,\n')
    f_out.write(',Q1,Q2,Q3,Q4,Q1,Q2,Q3,Q4\n')
    metrics = ['attribute_coverage', 'attribute_distribution']
    framed_attrs_role = ['Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds']
    framed_attrs_role_positions = ['Shooting Guard', 'Shooting Guard', 'Point Guard', 'Center', 'Power Forward', 'Small Forward', 'Power Forward']
    framed_attrs_size = ['Height (Inches)', 'Weight (Pounds)']
    all_positions = ['Center', 'Power Forward', 'Small Forward', 'Shooting Guard', 'Point Guard']
    for i in range(0, len(metrics)):
        for j in range(0, len(framed_attrs_role)):
            line = metrics[i] + ' - ' + framed_attrs_role[j] + '(' + framed_attrs_role_positions[j] + ')'
            size_quants = []
            role_quants = []
            for k in range(0, 4):
                size_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_size']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                role_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
            for k in range(0, 4):
                line += ',' + str(size_quants[k])
            for k in range(0, 4):
                line += ',' + str(role_quants[k])
            f_out.write(line + '\n')
            
        for j in range(0, len(framed_attrs_size)):
            for k in range(0, len(all_positions)):
                line = metrics[i] + ' - ' + framed_attrs_size[j] + '(' + all_positions[k] + ')'
                size_quants = []
                role_quants = []
                for l in range(0, 4):
                    size_quants.append(all_users_summary[all_positions[k]]['cond_size']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    role_quants.append(all_users_summary[all_positions[k]]['cond_role']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                for l in range(0, 4):
                    line += ',' + str(size_quants[l])
                for l in range(0, 4):
                    line += ',' + str(role_quants[l])
                f_out.write(line + '\n')
    f_out.close()
    
    # now separate out just the csv info we need for comparison by position
    f_out = open(bias_util.directory + 'analysis/csv/quantile_analysis_positions.csv', 'w+')
    f_out.write('metric,position,,,,non-position,,,\n')
    f_out.write(',Q1,Q2,Q3,Q4,Q1,Q2,Q3,Q4\n')
    metrics = ['attribute_coverage', 'attribute_distribution']
    framed_attrs_role = ['Avg. 3-Pointers Att.', 'Avg. 3-Pointers Made', 'Avg. Assists', 'Avg. Blocks', 'Avg. Offensive Rebounds', 'Avg. Steals', 'Avg. Total Rebounds']
    framed_attrs_role_positions = ['Shooting Guard', 'Shooting Guard', 'Point Guard', 'Center', 'Power Forward', 'Small Forward', 'Power Forward']
    framed_attrs_size = ['Height (Inches)', 'Weight (Pounds)']
    all_positions = ['Center', 'Power Forward', 'Small Forward', 'Shooting Guard', 'Point Guard']
    for i in range(0, len(metrics)):
        for j in range(0, len(framed_attrs_role)):
            line = metrics[i] + ' - ' + framed_attrs_role[j] + '(' + framed_attrs_role_positions[j] + ')'
            pos_quants = []
            non_pos_quants = []
            for k in range(0, 4):
                pos_quants.append(all_users_summary[framed_attrs_role_positions[j]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                cur_non_pos_quant = []
                for l in range(0, len(all_positions)):
                    if (all_positions[l] != framed_attrs_role_positions[j]):
                        cur_non_pos_quant.append(all_users_summary[all_positions[l]]['cond_role']['summary'][metrics[i]][framed_attrs_role[j]]['Q' + str(k + 1)]['mean'])
                non_pos_quants.append(np.mean(cur_non_pos_quant))
            for k in range(0, 4):
                line += ',' + str(pos_quants[k])
            for k in range(0, 4):
                line += ',' + str(non_pos_quants[k])
            f_out.write(line + '\n')
            
        for j in range(0, len(framed_attrs_size)):
            for k in range(0, len(all_positions)):
                line = metrics[i] + ' - ' + framed_attrs_size[j] + '(' + all_positions[k] + ')'
                pos_quants = []
                non_pos_quants = []
                for l in range(0, 4):
                    pos_quants.append(all_users_summary[all_positions[k]]['cond_size']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    cur_non_pos_quant = []
                    for m in range(0, len(all_positions)):
                        if (all_positions[m] != all_positions[k]):
                            cur_non_pos_quant.append(all_users_summary[all_positions[m]]['cond_role']['summary'][metrics[i]][framed_attrs_size[j]]['Q' + str(l + 1)]['mean'])
                    non_pos_quants.append(np.mean(cur_non_pos_quant))
                for l in range(0, 4):
                    line += ',' + str(pos_quants[l])
                for l in range(0, 4):
                    line += ',' + str(non_pos_quants[l])
                f_out.write(line + '\n')
    f_out.close()
    '''
    
    print '** Quantile Analysis Files Written'
    return all_users_summary
     
# do some of the post-hoc analyses on the data
if __name__ == '__main__': 
    '''fig_num = 2
    
    all_users = [f[5 :] for f in listdir(bias_util.directory) if ('user_' in f and not isfile(join(bias_util.directory, f)))]
    user_ids = []
    directories = []
    log_files = []
    for i in range(0, len(all_users)):
        cur_user = all_users[i]
        cur_dir = bias_util.directory + 'user_' + cur_user + '/'
        cur_file = 'interactions_' + cur_user + '.json'
        
        user_ids.append(cur_user)
        directories.append(cur_dir + 'logs/')
        log_files.append(cur_file)
        write_classification_accuracy(cur_dir + 'logs/', cur_file.replace('interactions', 'accuracy'), 'interactions_' + cur_user + '.json', fig_num + 1, bias_util.verbose) # fig_num = 3
   '''
    
    '''
    fig_num = 2
    # generate id-confusion matrices and svm results
    all_users = [f[5 :] for f in listdir(bias_util.directory) if ('user_' in f and not isfile(join(bias_util.directory, f)))]
    user_ids = []
    directories = []
    log_files = []
    for i in range(0, len(all_users)):
        cur_user = all_users[i]
        cur_dir = bias_util.directory + 'user_' + cur_user + '/'
        cur_file = 'interactions_' + cur_user + '.json'
        
        user_ids.append(cur_user)
        directories.append(cur_dir + 'logs/')
        log_files.append(cur_file)
        
        #interaction_prediction(cur_dir + 'logs/', 'interactions_' + cur_user + '.json', cur_user, bias_util.directory + 'analysis/', bias_util.verbose) # emily
        #classification_prediction(cur_dir + 'logs/', 'interactions_' + cur_user + '.json', bias_util.directory + 'analysis/', cur_user, bias_util.verbose)
        #break
        
        #write_id_confusion_matrix(cur_dir + 'logs/', cur_file.replace('interactions', 'id-conf'), 'interactions_' + cur_user + '.json', fig_num, bias_util.verbose) # fig_num = 2
        #write_classification_accuracy(cur_dir + 'logs/', cur_file.replace('interactions', 'accuracy'), 'interactions_' + cur_user + '.json', fig_num + 1, bias_util.verbose) # fig_num = 3
        #write_svm_results(cur_dir + 'logs/', cur_file.replace('interactions', 'svm'), 'interactions_' + cur_user + '.json', plot_svm, fig_num + 2, bias_util.verbose) # fig_num = 4 - 12
    '''
    
    
    # create decision trees to predict the labeling order differences between the two different conditions
    '''
    directories_var = [bias_util.directory + 'user_' + str(cur_user) + '/' + 'logs/' for cur_user in bias_util.cond_role]
    log_files_var = ['interactions_' + str(cur_user) + '.json' for cur_user in bias_util.cond_role]
    #classification_prediction(directories_var, log_files_var, bias_util.cond_role, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #interaction_prediction(directories_var, log_files_var, bias_util.cond_role, bias_util.directory + 'analysis/', bias_util.verbose) # emily

    directories_size = [bias_util.directory + 'user_' + str(cur_user) + '/' + 'logs/' for cur_user in bias_util.cond_size]
    log_files_size = ['interactions_' + str(cur_user) + '.json' for cur_user in bias_util.cond_size]
    classification_prediction(directories_size, log_files_size, bias_util.cond_size, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #interaction_prediction(directories_size, log_files_size, bias_util.cond_size, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    '''
    
    # get the list of axis changes for each user
    axis_changes(bias_util.all_participants)
    
    #interaction_prediction(directories, log_files, user_ids, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    #classification_prediction(directories, log_files, user_ids, bias_util.directory + 'analysis/', bias_util.verbose) # emily
    
    #out_dir = bias_util.directory + 'analysis/'
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
     
    #dtw_metric_map, all_users, all_metrics = dynamic_time_warping(True)
    # write the .csv file with the all DTW info  
    #write_dtw('/Users/emilywall/Desktop/', bias_util.verbose) #write_dtw(out_dir, bias_util.verbose)
    
    #quantile_analysis(bias_util.all_participants)
    #quantile_analysis_per_participant(bias_util.all_participants)