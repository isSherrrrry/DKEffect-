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

	new_data=[]


	for i in data:
		player_name = i['dataItem']['Name']
		data_loc = i['customLogInfo']['data_locations']
		quadrant = ''
		for q in data_loc:
			if q['player'] == player_name:
				if  (0 <= q['cx'] < 425) and (0 < q['cy'] <= 175):
					quadrant = 'A'
				elif (425 <= q['cx'] < 850) and (0 < q['cy'] <= 175):
					quadrant = 'B'
				elif (0 <= q['cx'] < 425) and (175 < q['cy'] <= 350):
					quadrant = 'C'
				else:
					quadrant = 'D'
		event_type = i['customLogInfo']['eventType']
		if(event_type == 'click'):
			duration = 0
		else:
			duration = i['customLogInfo']['elapsedTime'] 
		myjson_object = {
	                'Quadrant': quadrant,
	                'Event': event_type,
	                'Duration': duration
	            }
		new_data.append(myjson_object)

	quadrants=[]
	size = len(new_data) -1

	for i in new_data:
		quadrants.append(i['Quadrant'])

	first = []
	second =[]
	for i in range(len(quadrants) - 1):
		first.append(quadrants[i])
		second.append(quadrants[i+1])

	weight = [1]*size

	matrix_ad = pd.DataFrame({'source': first, "target": second, 'weight': weight})
	per = matrix_ad.groupby(["source","target"]).size().reset_index(name="weight")
	per.loc[per.source == per.target, 'weight'] = 0
	final= per.pivot_table(index='source',columns='target',values='weight')
	final = final.fillna(0)

	type_list = ['A', 'B', 'C','D']
	plot_data = final.reindex(type_list, axis="columns")
	plot_data_final = plot_data.reindex(type_list, axis="index")

	plot_data_final.to_csv('../Quadrant4/Quadrant4_Type1_Diagonal/Type1_4Quad_matrix_' + user_name +'.csv', encoding='utf-8')
	plt.figure(figsize=(20,10))
	ax = sns.heatmap(plot_data_final,cmap='Blues',xticklabels=True,yticklabels=True)
	plt.savefig('../Quadrant4/Quadrant4_Type1_Diagonal/Type1_Quad4_heatmap_' + user_name + '.png')

