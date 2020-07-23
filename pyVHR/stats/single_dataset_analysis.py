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
from nonparametric_tests import friedman_aligned_ranks_test as ft
import Orange

def sort_nicely(l): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    l.sort( key=alphanum_key ) 

    return l

#Dataset on which perform analysis
#DATASET = 'LGI-PPGI'
#DATASET = 'PURE'
DATASET = 'UBFC1'
#DATASET = 'UBFC2'
#DATASET = 'Cohface'
#DATASET = 'Mahnob'
#DATASET = 'UBFC_ALL'

CASE = 'full'
#CASE = 'split'

alpha = '0.05'

if DATASET == 'UBFC_ALL':
    exp_path1 = '../../results/' + 'UBFC1' + '/'
    files1 = sort_nicely(os.listdir(exp_path1))
    exp_path2 = '../../results/' + 'UBFC2' + '/'
    files2 = sort_nicely(os.listdir(exp_path2))
else:
    #Experiment Path
    exp_path = '../../results/' + DATASET + '/'
    files = sort_nicely(os.listdir(exp_path))

#All rPPG methods used
all_methods = ['CHROM','Green','ICA','LGI','PBV','PCA','POS','SSR']

#Method(s) for the visualization of the performance vs winSize
#methods = ['POS', 'CHROM', 'LGI']

#Metrics to Visualize
#metrics = ['CC', 'MAE', 'RMSE']
metrics = ['MAE']


print(all_methods)

#---------------- Produce Box plots for each method on a given dataset -----------

win_to_use = 10

if DATASET == 'UBFC_ALL':
    f_to_use = [i for i in files1 if 'winSize'+str(win_to_use) in i][0]
    path = exp_path1 + f_to_use
    res1 = pd.read_hdf(path)
    f_to_use = [i for i in files2 if 'winSize'+str(win_to_use) in i][0]
    path = exp_path2 + f_to_use
    res2 = pd.read_hdf(path)
    res = res1.append(res2)
else:
    f_to_use = [i for i in files if 'winSize'+str(win_to_use) in i][0]
    path = exp_path + f_to_use
    res = pd.read_hdf(path)

print('\n\n\t\t' + DATASET + '\n\n')

if DATASET == 'UBFC1' or DATASET == 'UBFC2' or DATASET == 'Mahnob' or DATASET == 'UBFC_ALL' or DATASET == 'Cohface':

    all_vals_CC = []
    all_vals_MAE = []
    all_vals_RMSE = []
    
    for metric in metrics:
        for method in all_methods:
            #print(method)
            mean_v = []
            raw_values = res[res['method'] == method][metric]
            print(raw_values)
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

              
    
    
    data_MAE = np.zeros([len(all_vals_MAE[0]), len(all_vals_MAE)])
    for i,m in enumerate(all_vals_MAE):
        data_MAE[:,i] = m
        
    print(data_MAE)
    
    '''data_MAE_df = pd.DataFrame(data_MAE, columns=all_methods)
    print('\nFriedman Test MAE:')
    print(ss.friedmanchisquare(*data_MAE.T))
    print(' ')'''

    '''pc = sp.posthoc_nemenyi_friedman(data_MAE_df)
    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    plt.figure()
    sp.sign_plot(pc, **heatmap_args)
    plt.title('Nemenyi Test MAE')'''

    n_datasets = data_MAE.shape[0]

    t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], 
                               data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
    avranksMAE = list(np.divide(ranks_mae, n_datasets))
    print('statistic: ' + str(t))
    print('pvalue: ' + str(p))
    print(' ')

    data_CC = np.zeros([len(all_vals_CC[0]), len(all_vals_CC)])
    for i,m in enumerate(all_vals_CC):
        data_CC[:,i] = m
    
    '''data_CC_df = pd.DataFrame(data_CC, columns=all_methods)
    print('\nFriedman Test MAE:')
    print(ss.friedmanchisquare(*data_CC.T))
    print(' ')
    pc = sp.posthoc_nemenyi_friedman(data_CC_df)
    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    plt.figure()
    sp.sign_plot(pc, **heatmap_args)
    plt.title('Nemenyi Test CC')'''

    t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], data_CC[:,4], 
                             data_CC[:,5], data_CC[:,6], data_CC[:,7])
    avranksCC = list(np.divide(ranks_cc, n_datasets))
    print('statistic: ' + str(t))
    print('pvalue: ' + str(p))
    print(' ')

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.title('CC')
    #plt.boxplot(all_vals_CC, showfliers=False)
    #plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

    #plt.subplot(1,2,2)
    #plt.title('MAE')
    #plt.boxplot(all_vals_MAE, showfliers=False)
    #plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

    cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
    Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
    #plt.title('CD Diagram MAE')

    cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
    Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
    #plt.title('CD Diagram CC')

    #plt.show()

elif DATASET == 'PURE':

    cases = {'01':'steady', '02':'talking', '03':'slow_trans', '04':'fast_trans', '05':'small_rot', '06':'fast_rot'}
    all_CC = {'01':[], '02':[], '03':[], '04':[], '05':[], '06':[]}
    all_MAE = {'01':[], '02':[], '03':[], '04':[], '05':[], '06':[]}

    if CASE == 'split':
        
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
            '''plt.figure()

            plt.subplot(1,2,1)
            plt.title('CC ' + cases[curr_case])
            plt.boxplot(all_CC[curr_case], showfliers=False)
            plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

            plt.subplot(1,2,2)
            plt.title('MAE ' + cases[curr_case])
            plt.boxplot(all_MAE[curr_case], showfliers=False)
            plt.xticks(np.arange(1,len(all_methods)+1), all_methods)'''

            print('\n' + curr_case + '\n')

            data_MAE = np.zeros([len(all_MAE[curr_case][0]), len(all_MAE[curr_case])])
            for i,m in enumerate(all_MAE[curr_case]):
                data_MAE[:,i] = m

            n_datasets = data_MAE.shape[0]

            data_CC = np.zeros([len(all_CC[curr_case][0]), len(all_CC[curr_case])])
            for i,m in enumerate(all_CC[curr_case]):
                data_CC[:,i] = m

            t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], 
                                       data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
            avranksMAE = list(np.divide(ranks_mae, n_datasets))
            print('statistic: ' + str(t))
            print('pvalue: ' + str(p))
            print(' ')

            t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], 
                                     data_CC[:,4], data_CC[:,5], data_CC[:,6], data_CC[:,7])
            avranksCC = list(np.divide(ranks_cc, n_datasets))
            print('statistic: ' + str(t))
            print('pvalue: ' + str(p))
            print(' ')

            cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
            Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
            plt.title('CD Diagram MAE')

            cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
            Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
            plt.title('CD Diagram CC')

        plt.show()
    
    elif CASE == 'full':
        
        CC_allcases = []
        MAE_allcases = []
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
        
        data_MAE = np.zeros([len(MAE_allcases[0]), len(MAE_allcases)])
        for i,m in enumerate(MAE_allcases):
            data_MAE[:,i] = m
        
        n_datasets = data_MAE.shape[0]

        '''data_MAE_df = pd.DataFrame(data_MAE, columns=all_methods)
        print('\nFriedman Test MAE:')
        print(ss.friedmanchisquare(*data_MAE.T))
        print(' ')
        pc = sp.posthoc_nemenyi_friedman(data_MAE_df)
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        plt.figure()
        sp.sign_plot(pc, **heatmap_args)
        plt.title('Nemenyi Test MAE')'''

        t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
        avranksMAE = list(np.divide(ranks_mae, n_datasets))
        print('statistic: ' + str(t))
        print('pvalue: ' + str(p))
        print(' ')

        data_CC = np.zeros([len(CC_allcases[0]), len(CC_allcases)])
        for i,m in enumerate(CC_allcases):
            data_CC[:,i] = m
        
        '''data_CC_df = pd.DataFrame(data_CC, columns=all_methods)
        print('\nFriedman Test MAE:')
        print(ss.friedmanchisquare(*data_CC.T))
        print(' ')
        pc = sp.posthoc_nemenyi_friedman(data_CC_df)
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        plt.figure()
        sp.sign_plot(pc, **heatmap_args)
        plt.title('Nemenyi Test CC')'''

        t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], data_CC[:,4], data_CC[:,5], data_CC[:,6], data_CC[:,7])
        avranksCC = list(np.divide(ranks_cc, n_datasets))
        print('statistic: ' + str(t))
        print('pvalue: ' + str(p))
        print(' ')

        '''plt.figure()
        plt.subplot(1,2,1)
        plt.title('CC')
        plt.boxplot(CC_allcases, showfliers=False)
        plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

        plt.subplot(1,2,2)
        plt.title('MAE')
        plt.boxplot(MAE_allcases, showfliers=False)
        plt.xticks(np.arange(1,len(all_methods)+1), all_methods)'''

        cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
        Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
        plt.title('CD Diagram MAE')

        cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
        Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
        plt.title('CD Diagram CC')

        plt.show()


elif DATASET == 'LGI-PPGI':

    cases = ['gym', 'resting', 'rotation', 'talk']
    all_CC = {'gym':[], 'resting':[], 'rotation':[], 'talk':[]}
    all_MAE = {'gym':[], 'resting':[], 'rotation':[], 'talk':[]}

    if CASE == 'split':

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
            
            plt.figure()

            plt.subplot(1,2,1)
            plt.title('CC ' + curr_case)
            plt.boxplot(all_CC[curr_case], showfliers=False)
            plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

            plt.subplot(1,2,2)
            plt.title('MAE ' + curr_case)
            plt.boxplot(all_MAE[curr_case], showfliers=False)
            plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

            print('\n' + curr_case + '\n')

            data_MAE = np.zeros([len(all_MAE[curr_case][0]), len(all_MAE[curr_case])])
            for i,m in enumerate(all_MAE[curr_case]):
                data_MAE[:,i] = m

            n_datasets = data_MAE.shape[0]

            data_CC = np.zeros([len(all_CC[curr_case][0]), len(all_CC[curr_case])])
            for i,m in enumerate(all_CC[curr_case]):
                data_CC[:,i] = m

            t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
            avranksMAE = list(np.divide(ranks_mae, n_datasets))
            print('statistic: ' + str(t))
            print('pvalue: ' + str(p))
            print(' ')

            t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], data_CC[:,4], data_CC[:,5], data_CC[:,6], data_CC[:,7])
            avranksCC = list(np.divide(ranks_cc, n_datasets))
            print('statistic: ' + str(t))
            print('pvalue: ' + str(p))
            print(' ')

            cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
            Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
            plt.title('CD Diagram MAE')

            cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
            Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
            plt.title('CD Diagram CC')

        plt.show()

    elif CASE == 'full':
        
        CC_allcases = []
        MAE_allcases = []
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
        
        data_MAE = np.zeros([len(MAE_allcases[0]), len(MAE_allcases)])
        for i,m in enumerate(MAE_allcases):
            data_MAE[:,i] = m

        n_datasets = data_MAE.shape[0]
        
        data_MAE_df = pd.DataFrame(data_MAE, columns=all_methods)
        print('\nFriedman Test MAE:')
        print(ss.friedmanchisquare(*data_MAE.T))
        print(' ')
        pc = sp.posthoc_nemenyi_friedman(data_MAE_df)
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        plt.figure()
        sp.sign_plot(pc, **heatmap_args)
        plt.title('Nemenyi Test MAE')

        t,p,ranks_mae,piv_mae = ft(data_MAE[:,0], data_MAE[:,1], data_MAE[:,2], data_MAE[:,3], data_MAE[:,4], data_MAE[:,5], data_MAE[:,6], data_MAE[:,7])
        avranksMAE = list(np.divide(ranks_mae, n_datasets))
        print('statistic: ' + str(t))
        print('pvalue: ' + str(p))
        print(' ')

        data_CC = np.zeros([len(CC_allcases[0]), len(CC_allcases)])
        for i,m in enumerate(CC_allcases):
            data_CC[:,i] = m
        
        data_CC_df = pd.DataFrame(data_CC, columns=all_methods)
        print('\nFriedman Test CC:')
        print(ss.friedmanchisquare(*data_CC.T))
        print(' ')
        pc = sp.posthoc_nemenyi_friedman(data_CC_df)
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        plt.figure()
        sp.sign_plot(pc, **heatmap_args)
        plt.title('Nemenyi Test CC')

        t,p,ranks_cc,piv_cc = ft(data_CC[:,0], data_CC[:,1], data_CC[:,2], data_CC[:,3], data_CC[:,4], data_CC[:,5], data_CC[:,6], data_CC[:,7])
        avranksCC = list(np.divide(ranks_cc, n_datasets))
        print('statistic: ' + str(t))
        print('pvalue: ' + str(p))
        print(' ')

        plt.figure()
        plt.subplot(1,2,1)
        plt.title('CC')
        plt.boxplot(CC_allcases, showfliers=False)
        plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

        plt.subplot(1,2,2)
        plt.title('MAE')
        plt.boxplot(MAE_allcases, showfliers=False)
        plt.xticks(np.arange(1,len(all_methods)+1), all_methods)

        cd = Orange.evaluation.compute_CD(avranksMAE, n_datasets, alpha=alpha) #tested on 30 datasets
        Orange.evaluation.graph_ranks(avranksMAE, all_methods, cd=cd, width=6, textspace=1.5, reverse=True)
        plt.title('CD Diagram MAE')

        cd = Orange.evaluation.compute_CD(avranksCC, n_datasets, alpha=alpha) #tested on 30 datasets
        Orange.evaluation.graph_ranks(avranksCC, all_methods, cd=cd, width=6, textspace=1.5)
        plt.title('CD Diagram CC')

        plt.show()
