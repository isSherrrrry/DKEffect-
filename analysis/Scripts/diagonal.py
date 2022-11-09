import json
import glob
import os
import os.path
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

msci_folder = '../Clean_Files'
mscifile = glob.glob(msci_folder + '\*.json')
for file in mscifile:
	p = Path(file)
	user_name = p.name[13:26]
	with open(file) as json_file:
		data = json.load(json_file)

	players=[]
	size = len(data) -1

	for i in data:
		players.append(i['dataItem']['Name'])


	first = []
	second =[]
	for i in range(len(players) - 1):
		first.append(players[i])
		second.append(players[i+1])

	weight = [1]*size

	matrix_ad = pd.DataFrame({'source': first, "target": second, 'weight': weight})
	matrix_ad.to_csv('tablesd.csv', encoding='utf-8')

	per = matrix_ad.groupby(["source","target"]).size().reset_index(name="weight")
	per.to_csv('tablegroup.csv', encoding='utf-8')

	per.loc[per.source == per.target, 'weight'] = 0
	per.to_csv('table_diagonal.csv', encoding='utf-8')

	final= per.pivot_table(index='source',columns='target',values='weight')
	final.to_csv('final_diagonal.csv', encoding='utf-8')

	#final = final.fillna(0)

	type_list = ['Player 47', 'Player 114', 'Player 127', 'Player 136', 'Player 166', 'Player 299', 'Player 310',
	 'Player 324', 'Player 348', 'Player 383', 'Player 384', 'Player 394', 'Player 463', 'Player 473',
	  'Player 597', 'Player 603', 'Player 611', 'Player 621', 'Player 852', 'Player 903', 'Player 51',
	   'Player 79', 'Player 86', 'Player 134', 'Player 195', 'Player 213', 'Player 309', 'Player 433',
	    'Player 496', 'Player 513', 'Player 571', 'Player 600', 'Player 702', 'Player 752', 'Player 771',
	     'Player 799', 'Player 859', 'Player 920', 'Player 942', 'Player 966', 'Player 23', 'Player 107',
	      'Player 113', 'Player 128', 'Player 161', 'Player 170', 'Player 335', 'Player 520', 'Player 562', 
	      'Player 606', 'Player 678', 'Player 686', 'Player 762', 'Player 819', 'Player 824', 'Player 854',
	       'Player 878', 'Player 959', 'Player 980', 'Player 995', 'Player 8', 'Player 11', 'Player 13', 'Player 30',
	        'Player 49', 'Player 58', 'Player 162', 'Player 167', 'Player 253', 'Player 339', 'Player 391', 
	        'Player 442', 'Player 554', 'Player 570', 'Player 763', 'Player 900', 'Player 919', 'Player 921',
	     'Player 952', 'Player 992', 'Player 85', 'Player 185', 'Player 199', 'Player 278', 'Player 289', 
	     'Player 380', 'Player 395', 'Player 537', 'Player 574', 'Player 585', 'Player 590', 'Player 640', 
	     'Player 689', 'Player 751', 'Player 755', 'Player 764', 'Player 828', 'Player 884', 'Player 887', 'Player 894']


	plot_data = final.reindex(type_list, axis="columns")
	plot_data_final = plot_data.reindex(type_list, axis="index")
	plot_data_final = plot_data_final.fillna(0)
	plot_data.to_csv('../Type1_Matrix/with_order_diagonal/Type1_matrix_' + user_name +'.csv', encoding='utf-8')

	plt.figure(figsize=(20,10))
	ax = sns.heatmap(plot_data_final,cmap='Blues',xticklabels=True,yticklabels=True)
	plt.savefig('../Type1_Matrix/with_order_diagonal/Type1_heatmap_' + user_name + '.png')