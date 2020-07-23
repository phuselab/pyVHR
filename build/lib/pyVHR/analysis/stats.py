import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as ss
import scikit_posthocs as sp
from .stattests import friedman_aligned_ranks_test as ft
import Orange

class StatAnalysis():
    """ Statistics analysis for multiple datasets and multiple VHR methods"""
    
    def __init__(self, filepath='default'):
        
        if os.path.isdir(filepath):  
            self.multidataset = True
            self.path = filepath + "/"
            self.datasetsList = os.listdir(filepath)
        elif os.path.isfile(filepath):  
            self.multidataset = False
            self.datasetsList = [filepath]
            self.path = ""
        else:  
            raise("Error: filepath is wrong!")
        
        # -- get common methods
        self.__getMethods()
        self.metricSort = {'MAE':'min','RMSE':'min','CC':'max','PCC':'max'}
        self.scale = {'MAE':'log','RMSE':'log','CC':'linear','PCC':'linear'}

    def FriedmanTest(self, methods=None, metric='MAE'):
        
        # -- Method(s) 
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise("Some method is wrong!")
            else:
                self.methods = methods
           
        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        
        # -- get data from dataset(s)
        #    return Y = mat(n-datasets,k-methods)   
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
        self.ndataset = Y.shape[0]
        
        # -- Friedman test
        t,p,ranks,piv = ft(Y)
        self.avranks = list(np.divide(ranks, self.ndataset))
        
        return t,p,ranks,piv,self.ndataset
    
    def SignificancePlot(self, methods=None, metric='MAE'):

        # -- Method(s) 
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise("Some method is wrong!")
            else:
                self.methods = methods
           
        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        
        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
        
        # -- Significance plot, a heatmap of p values
        methodNames = [x.upper() for x in self.methods]
        Ypd = pd.DataFrame(Y, columns=methodNames)
        ph = sp.posthoc_nemenyi_friedman(Ypd)
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 
                        'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}

        plt.figure(figsize=(5,4))
        sp.sign_plot(ph, cbar=True, **heatmap_args)
        plt.title('p-vals')
        
        fname = 'SP_' + self.metric + '.pdf'
        plt.savefig(fname)
        plt.show()
        
    def computeCD(self, avranks=None, numDatasets=None, alpha='0.05', display=True):
        """
        Returns critical difference for Nemenyi or Bonferroni-Dunn test according 
        to given alpha (either alpha=”0.05” or alpha=”0.1”) for average ranks and 
        number of tested datasets N. Test can be either “nemenyi” for for Nemenyi 
        two tailed test or “bonferroni-dunn” for Bonferroni-Dunn test.
        See Orange package docs.
        """
        if not numDatasets:
            numDatasets = self.ndataset
        if not avranks:
            avranks = self.avranks
        
        cd = Orange.evaluation.compute_CD(avranks, numDatasets, alpha=alpha) #tested on 30 datasets
        
        if self.mag == 'min':
            reverse = True
        else:
            reverse = False
        
        methodNames = [x.upper() for x in self.methods]
        if display:
            Orange.evaluation.graph_ranks(avranks, methodNames, cd=cd, width=6, textspace=1.5, reverse=reverse)
            name = 'CD Diagram (metric: ' + self.metric +')'
            plt.title(name)  
            fname = 'CD_' + self.metric + '.pdf'
            plt.savefig(fname)
            
            plt.show()
        return cd

    def displayBoxPlot(self, methods=None, metric='MAE', scale=None, title=True):
  
        # -- Method(s) 
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise("Some method is wrong!")
            else:
                self.methods = methods
           
        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        if scale == None:
            scale = self.scale[metric]
        
        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
       
        # -- display box plot
        self.boxPlot(methods, metric, Y, scale=scale, title=title)
        
    def boxPlot(self, methods, metric, Y, scale, title):
        
        #  Y = mat(n-datasets,k-methods)   
        
        k = len(methods)
        
        if not (k == Y.shape[1]):
            raise("error!")

        offset = 50
        fig = go.Figure()

        methodNames = [x.upper() for x in self.methods]
        for i in range(k):
            yd = Y[:,i]
            name = methodNames[i]
            # -- set color for box
            if metric == 'MAE' or  metric == 'RMSE':
                med = np.median(yd)
                col = str(min(200,5*int(med)+offset))
            if metric == 'CC' or metric == 'PCC':
                med = 1-np.abs(np.median(yd))
                col = str(int(200*med)+offset)

            # -- add box 
            fig.add_trace(go.Box(
                y=yd,
                name=name,
                boxpoints='all',
                jitter=.7,
                #whiskerwidth=0.2,
                fillcolor="rgba("+col+","+col+","+col+",0.5)",
                line_color="rgba(0,0,255,0.5)",
                marker_size=2,
                line_width=2)
            )

        gwidth = np.max(Y)/10
        
        if title:
            tit = "Metric: " + metric
            top = 40
        else:
            tit=''
            top = 10
        
        fig.update_layout(
            title=tit,
            yaxis_type=scale,
            xaxis_type="category",
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                #dtick=gwidth,
                gridcolor='rgb(255,255,255)',
                gridwidth=.1,
                zerolinewidth=2,
                titlefont=dict(size=30)
            ),
            font=dict(
                family="monospace",
                size=16,
                color='rgb(20,20,20)'
            ),
            margin=dict(
                l=20,
                r=10,
                b=20,
                t=top,
            ),
            paper_bgcolor='rgb(250, 250, 250)',
            plot_bgcolor='rgb(243, 243, 243)',
            showlegend=False
        )

        fig.show()
        
    def saveStatsData(self, methods=None, metric='MAE', outfilename='statsData.csv'):
        Y = self.getStatsData(methods=methods, metric=metric, printTable=False)
        np.savetxt(outfilename, Y)
        
    def getStatsData(self, methods=None, metric='MAE', printTable=True):
         # -- Method(s) 
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise("Some method is wrong!")
            else:
                self.methods = methods
           
        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]
        
          # -- get data from dataset(s)
        #    return Y = mat(n-datasets,k-methods)   
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
           
        # -- add median and IQR
        I = ss.iqr(Y,axis=0)
        M = np.median(Y,axis=0)
        Y = np.vstack((Y,M))
        Y = np.vstack((Y,I))
        
        if printTable:
            methodNames = [x.upper() for x in self.methods]
            dataseNames = self.datasetNames
            dataseNames.append('Median')
            dataseNames.append('IQR')
            df = pd.DataFrame(Y, columns=methodNames, index=dataseNames)
            display(df)
        
        return Y

    def __getDataMono(self):
        mag = self.mag
        metric = self.metric
        methods = self.methods
        
        frame = self.dataFrame[0]
        # -- loop on methods
        Y = []
        for method in methods:
            vals = frame[frame['method'] == method][metric]
            if mag == 'min':
                data = [v[np.argmin(v)] for v in vals]
            else:
                data = [v[np.argmax(v)] for v in vals]
            Y.append(data)
            
        return np.array(Y).T

    def __getData(self):
        
        mag = self.mag
        metric = self.metric
        methods = self.methods
        
        # -- loop on datasets
        Y = []
        for frame in self.dataFrame:
           
            # -- loop on methods
            y = []
            for method in methods:
                vals = frame[frame['method'] == method][metric]
                if mag == 'min':
                    data = [v[np.argmin(v)] for v in vals]
                else:
                    data = [v[np.argmax(v)] for v in vals]

                y.append(data)
            
            y = np.array(y)
            Y.append(np.mean(y,axis=1)) 
        return np.array(Y)
    
    def __getMethods(self):
        
        mets = []
        dataFrame = []
        N = len(self.datasetsList)
        
        # -- load dataframes
        self.datasetNames = []
        for file in self.datasetsList:
            filename = self.path + file
            self.datasetNames.append(file)
            data = pd.read_hdf(filename)
            mets.append(set(list(data['method'])))
            dataFrame.append(data)

        # -- method names intersection among datasets
        methods = set(mets[0])
        if N > 1:
            for m in range(1,N-1):
                methods.intersection(mets[m])

        methods = list(methods)
        methods.sort()
        self.methods = methods
        self.dataFrame = dataFrame

            
