import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import scipy.stats as ss
import scikit_posthocs as sp
import pandas as pd

def sort_nicely(l): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    l.sort( key=alphanum_key ) 

    return l


DATASETS = ['PURE', 'UBFC1', 'UBFC2', 'LGI-PPGI']
#DATASETS = ['PURE', 'UBFC1', 'UBFC2', 'LGI-PPGI', 'Cohface', 'Mahnob']
#DATASETS = ['Cohface', 'Mahnob']
all_methods = ['CHROM','Green','ICA','LGI','PBV','PCA','POS','SSR']
metrics = ['CC', 'MAE']

avg_type = 'mean'
#avg_type = 'median'

data_CC = []
data_MAE = []

for r,DATASET in enumerate(DATASETS):
    
    #Experiment Path
    exp_path = '../results/' + DATASET + '/'
    files = sort_nicely(os.listdir(exp_path))

    #---------------- Produce Box plots for each method on a given dataset -----------

    win_to_use = 10

    f_to_use = [i for i in files if 'winSize'+str(win_to_use) in i][0]
    path = exp_path + f_to_use
    res = pd.read_hdf(path)

    print('\n\n\t\t' + DATASET + '\n\n')

    if DATASET == 'UBFC1' or DATASET == 'UBFC2' or DATASET == 'Mahnob' or DATASET == 'UBFC_ALL':

        all_vals_CC = []
        all_vals_MAE = []
        curr_dataCC = np.zeros(len(all_methods))
        curr_dataMAE = np.zeros(len(all_methods))
        
        for metric in metrics:
            for method in all_methods:
                #print(method)
                mean_v = []
                raw_values = res[res['method'] == method][metric]
                values = []
                for v in raw_values:
                    if metric == 'CC':
                        values.append(v[np.argmax(v)])
                    else:
                        values.append(v[np.argmin(v)])

                if metric == 'CC':
                    all_vals_CC.append(np.array(values))
                if metric == 'MAE':
                    all_vals_MAE.append(np.array(values))

        for c in range(len(all_vals_CC)): #for each method
            if avg_type == 'median':
                curr_dataCC[c] = np.median(all_vals_CC[c])
                curr_dataMAE[c] = np.median(all_vals_MAE[c])
            else:
                curr_dataCC[c] = np.mean(all_vals_CC[c])
                curr_dataMAE[c] = np.mean(all_vals_MAE[c])

        data_CC.append(curr_dataCC)
        data_MAE.append(curr_dataMAE)
        

    elif DATASET == 'PURE':

        cases = {'01':'steady', '02':'talking', '03':'slow_trans', '04':'fast_trans', '05':'small_rot', '06':'fast_rot'}
        all_CC = {'01':[], '02':[], '03':[], '04':[], '05':[], '06':[]}
        all_MAE = {'01':[], '02':[], '03':[], '04':[], '05':[], '06':[]}
        CC_allcases = []
        MAE_allcases = []
        curr_dataCC = np.zeros(len(all_methods))
        curr_dataMAE = np.zeros(len(all_methods))
            
        for metric in metrics:
            for method in all_methods:
                #print(method)
                for curr_case in cases.keys():
                    
                    curr_res = res[res['videoName'].str.split('/').str[5].str.split('-').str[1] == curr_case]
                    raw_values = curr_res[curr_res['method'] == method][metric]
                    
                    values = []
                    for v in raw_values:
                        if metric == 'CC':
                            values.append(v[np.argmax(v)])
                        else:
                            values.append(v[np.argmin(v)])

                    if metric == 'CC':
                        all_CC[curr_case].append(np.array(values))
                    if metric == 'MAE':
                        all_MAE[curr_case].append(np.array(values))
        
        for curr_case in cases.keys():

            all_vals_CC = all_CC[curr_case]
            all_vals_MAE = all_MAE[curr_case]

            for c in range(len(all_vals_CC)): #for each method
                if avg_type == 'median':
                    curr_dataCC[c] = np.median(all_vals_CC[c])
                    curr_dataMAE[c] = np.median(all_vals_MAE[c])
                else:
                    curr_dataCC[c] = np.mean(all_vals_CC[c])
                    curr_dataMAE[c] = np.mean(all_vals_MAE[c])

            data_CC.append(curr_dataCC.copy())
            data_MAE.append(curr_dataMAE.copy())

    elif DATASET == 'Cohface':
            
        CC_allcases = []
        MAE_allcases = []
        curr_dataCC = np.zeros(len(all_methods))
        curr_dataMAE = np.zeros(len(all_methods))

        for metric in metrics:
            for method in all_methods:
                    raw_values = res[res['method'] == method][metric]
                    
                    values = []
                    for v in raw_values:
                        if metric == 'CC':
                            values.append(v[np.argmax(v)])
                        else:
                            values.append(v[np.argmin(v)])
                    
                    if metric == 'CC':
                        CC_allcases.append(np.array(values))
                    if metric == 'MAE':
                        MAE_allcases.append(np.array(values))

        for c in range(len(CC_allcases)): #for each method
            if avg_type == 'median':
                curr_dataCC[c] = np.median(all_vals_CC[c])
                curr_dataMAE[c] = np.median(all_vals_MAE[c])
            else:
                curr_dataCC[c] = np.mean(CC_allcases[c])
                curr_dataMAE[c] = np.mean(MAE_allcases[c])

        data_CC.append(curr_dataCC)
        data_MAE.append(curr_dataMAE)


    elif DATASET == 'LGI-PPGI':

        cases = ['gym', 'resting', 'rotation', 'talk']
        #cases = ['resting']
        all_CC = {'gym':[], 'resting':[], 'rotation':[], 'talk':[]}
        all_MAE = {'gym':[], 'resting':[], 'rotation':[], 'talk':[]}
        CC_allcases = []
        MAE_allcases = []
        curr_dataCC = np.zeros(len(all_methods))
        curr_dataMAE = np.zeros(len(all_methods))

        for metric in metrics:
            for method in all_methods:
                #print(method)
                for curr_case in cases:

                    curr_res = res[res['videoName'].str.split('/').str[6].str.split('_').str[1] == curr_case]
                    raw_values = curr_res[curr_res['method'] == method][metric]
                    
                    values = []
                    for v in raw_values:
                        if metric == 'CC':
                            values.append(v[np.argmax(v)])
                        else:
                            values.append(v[np.argmin(v)])

                    if metric == 'CC':
                        all_CC[curr_case].append(np.array(values))
                    if metric == 'MAE':
                        all_MAE[curr_case].append(np.array(values))

        for curr_case in cases:
            all_vals_CC = all_CC[curr_case]
            all_vals_MAE = all_MAE[curr_case]

            for c in range(len(all_vals_CC)): #for each method
                if avg_type == 'median':
                    curr_dataCC[c] = np.median(all_vals_CC[c])
                    curr_dataMAE[c] = np.median(all_vals_MAE[c])
                else:
                    curr_dataCC[c] = np.mean(all_vals_CC[c])
                    curr_dataMAE[c] = np.mean(all_vals_MAE[c])

            data_CC.append(curr_dataCC.copy())
            data_MAE.append(curr_dataMAE.copy())


data_CC = np.vstack(data_CC)
data_MAE = np.vstack(data_MAE)
n_datasets = data_CC.shape[0]
alpha = '0.05'

plt.figure()

plt.subplot(1,2,1)
plt.title('CC Multi Dataset')
plt.boxplot(data_CC, showfliers=True)
plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

plt.subplot(1,2,2)
plt.title('MAE Multi Dataset')
plt.boxplot(data_MAE, showfliers=True)
plt.xticks(np.arange(1,len(all_methods)+1), all_methods)
            
from nonparametric_tests import friedman_aligned_ranks_test as ft
import Orange

data_MAE_df = pd.DataFrame(data_MAE, columns=all_methods)
print('\nFriedman Test MAE:')
#print(ss.friedmanchisquare(*data_MAE.T))
#print(' ')
t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
avranksMAE = list(np.divide(ranks_mae, n_datasets))

print('statistic: ' + str(t))
print('pvalue: ' + str(p))
print(' ')
pc = sp.posthoc_nemenyi_friedman(data_MAE_df)
cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

plt.figure()
sp.sign_plot(pc, **heatmap_args)
plt.title('Nemenyi Test MAE')


data_CC_df = pd.DataFrame(data_CC, columns=all_methods)
print('\nFriedman Test CC:')
#print(ss.friedmanchisquare(*data_CC.T))
#print(' ')
t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], data_CC[:,4], data_CC[:,5], data_CC[:,6], data_CC[:,7])
avranksCC = list(np.divide(ranks_cc, n_datasets))

print('statistic: ' + str(t))
print('pvalue: ' + str(p))
print(' ')
pc = sp.posthoc_nemenyi_friedman(data_CC_df)
cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

plt.figure()
sp.sign_plot(pc, **heatmap_args)
plt.title('Nemenyi Test CC')

cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
plt.title('CD Diagram MAE')

cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
plt.title('CD Diagram CC')

print(data_MAE_df)
print(' ')
print(data_CC_df)

plt.show()