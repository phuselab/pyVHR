import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as ss
import scikit_posthocs as sp
from autorank import autorank, plot_stats, create_report
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase, Colorbar
import itertools


class StatAnalysis:
    """ 
    Statistic analyses for multiple datasets and multiple rPPG methods
    """

    def __init__(self, filepath, join_data=False, remove_outliers=False):
        """
        Args:
            filepath:
                - The path to the file contaning the results to test
            join_data: 
                - 'True' - If filepath is a folder, join the dataframes contained in the folder (to be used when wanting to merge multiple results from the same pipeline on the same dataset)
                - 'False' - (default) To be used if you want to test the same pipeline (eventually with multiple methods) on multiple datasets
            remove_outliers:
                - 'True' -  Remove outliers from data prior to statistical testing
                - 'False' - (default) no outlier removal
        """



        if os.path.isdir(filepath):
            self.multidataset = True
            self.path = filepath + "/"
            self.datasetsList = os.listdir(filepath)
        elif os.path.isfile(filepath):
            self.multidataset = False
            self.datasetsList = [filepath]
            self.path = ""
        else:
            raise "Error: filepath is wrong!"

        self.join_data = join_data
        self.available_metrics = ['MAE', 'RMSE', 'PCC', 'CCC', 'SNR']
        self.remove_outliers = remove_outliers

        # -- get data
        self.__getMethods()
        self.metricSort = {'MAE': 'min', 'RMSE': 'min', 'PCC': 'max', 'CCC': 'max', 'SNR': 'max'}
        self.scale = {'MAE': 'log', 'RMSE': 'log', 'PCC': 'linear', 'CCC': 'linear', 'SNR': 'linear'}

        self.use_stats_pipeline = False

    def __any_equal(self, mylist):
        equal = []
        for a, b in itertools.combinations(mylist, 2):
            equal.append(a == b)
        return np.any(equal)

    def run_stats(self, methods=None, metric='CCC', approach='frequentist', print_report=True):
        """
        Runs the statistical testing procedure by automatically selecting the appropriate test for the available data.

        Args:
            methods:
                - The rPPG methods to analyze
            metric:
                - 'MAE' - Mean Absolute Error
                - 'RMSE' - Root Mean Squared Error
                - 'PCC' - Pearson's Correlation Coefficient
                - 'CCC' - Concordance Correlation Coefficient
                - 'SNR' - Signal to Noise Ratio
            approach:
                - 'frequentist' - (default) Use frequentist hypotesis tests for the analysis
                - 'bayesian' - Use bayesian hypotesis tests for the analysis
            print_report:
                - 'True' - (default) print a report of the hypotesis testing procedure
                - 'False' - Doesn't print any report

        Returns:
            Y_df: A pandas DataFrame containing the data on which the statistical analysis has been performed
            fig: A matplotlib figure displaying the outcome of the statistical analysis (an empty figure if the wilcoxon test has been chosen)
        """
        metric = metric.upper()
        assert metric in self.available_metrics, 'Error! Available metrics are ' + str(self.available_metrics)

        # -- Method(s) 
        if methods is not None:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
            else:
                self.methods = methods

        assert approach == 'frequentist' or approach == 'bayesian', "Approach should be 'frequentist' or bayesian, not " + str(approach)

        # -- set metric
        self.metric = metric
        self.mag = self.metricSort[metric]

        # -- get data from dataset(s)
        if self.multidataset:
            Y = self.__getData()
        else:
            Y = self.__getDataMono()
        self.ndataset = Y.shape[0]

        if metric == 'MAE' or metric == 'RMSE':
            order = 'ascending'
        else:
            order = 'descending'

        m_names = [x.upper().replace('CUPY_', '').replace('CPU_', '').replace('TORCH_', '') for x in self.methods]
        if self.__any_equal(m_names):
            m_names = self.methods
        Y_df = pd.DataFrame(Y, columns=m_names)

        results = autorank(Y_df, alpha=0.05, order=order, verbose=False, approach=approach)
        self.stat_result = results
        self.use_stats_pipeline = True

        if approach == 'bayesian':
            res_df = results.rankdf.iloc[:, [0, 1, 4, 5, 8]]
            print(res_df)

        if print_report:
            print(' ')
            create_report(results)
            print(' ')

        fig = plt.figure(figsize=(12, 5))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
        _, ax = self.computeCD(approach=approach, ax=ax)

        return Y_df, fig


    def SignificancePlot(self, methods=None, metric='MAE'):
        """
        Returns a significance plot of the results of hypotesis testing
        """

        # -- Method(s) 
        if methods == None:
            methods = self.methods
        else:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
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
        methodNames = [x.upper().replace('CUPY_', '').replace('CPU_', '').replace('TORCH_', '') for x in self.methods]
        if self.__any_equal(methodNames):
            methodNames = self.methods
        Ypd = pd.DataFrame(Y, columns=methodNames)
        ph = sp.posthoc_nemenyi_friedman(Ypd)
        cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                        'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}

        fig = plt.figure(figsize=(10, 7))
        ax, cbar = sp.sign_plot(ph, cbar=True, **heatmap_args)
        ax.set_title('p-vals')
        return fig

    def computeCD(self, ax=None, avranks=None, numDatasets=None, alpha='0.05', display=True, approach='frequentist'):
        """
        Returns critical difference and critical difference diagram for Nemenyi post-hoc test if the frequentist approach has been chosen
        Returns a Plot of the results of bayesian significance testing otherwise
        """
        cd = self.stat_result.cd
        if display and approach == 'frequentist':
            stats_fig = plot_stats(self.stat_result, allow_insignificant=True, ax=ax)
        elif display and approach == 'bayesian':
            stats_fig = self.plot_bayesian_res(self.stat_result)
        return cd, stats_fig

    def plot_bayesian_res(self, stat_result):
        """
        Plots the results of bayesian significance testing
        """
        dm = stat_result.decision_matrix.copy()
        cmap = ['1', '#fb6a4a', '#08306b', '#4292c6']  # , '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5',
                        'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}
        dm[dm == 'inconclusive'] = 0
        dm[dm == 'smaller'] = 1
        dm[dm == 'larger'] = 2
        np.fill_diagonal(dm.values, -1)

        pl, ax = plt.subplots()
        ax.imshow(dm.values.astype(int), cmap=ListedColormap(cmap))
        labels = list(dm.columns)
        # Major ticks
        ax.set_xticks(np.arange(0, len(labels), 1))
        ax.set_yticks(np.arange(0, len(labels), 1))
        # Labels for major ticks
        ax.set_xticklabels(labels, rotation='vertical')
        ax.set_yticklabels(labels)
        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.set_title('Metric: ' + self.metric)
        cbar_ax = ax.figure.add_axes([0.85, 0.35, 0.04, 0.3])
        cbar = ColorbarBase(cbar_ax, cmap=ListedColormap(cmap), boundaries=[0, 1, 2, 3, 4])
        cbar.set_ticks(np.linspace(0.5, 3.5, 4))
        cbar.set_ticklabels(['None', 'equivalent', 'smaller', 'larger'])
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor('0.5')
        cbar.ax.tick_params(size=0)
        return pl

    def displayBoxPlot(self, methods=None, metric='MAE', scale=None, title=True):
        """
        Shows the distribution of populations with box-plots 
        """
        metric = metric.upper()

        # -- Method(s) 
        if methods is None:
            methods = self.methods
        else:
            if not set(methods).issubset(set(self.methods)):
                raise ValueError("Some method is wrong!")
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
        fig = self.boxPlot(methods, metric, Y, scale=scale, title=title)

        return fig

    def boxPlot(self, methods, metric, Y, scale, title):
        """
        Creates the box plot 
        """

        #  Y = mat(n-datasets,k-methods)   

        k = len(methods)

        if not (k == Y.shape[1]):
            raise ("error!")

        offset = 50
        fig = go.Figure()

        methodNames = [x.upper().replace('CUPY_', '').replace('CPU_', '').replace('TORCH_', '') for x in methods]
        if self.__any_equal(methodNames):
            methodNames = methods
        for i in range(k):
            yd = Y[:, i]
            name = methodNames[i]
            if len(np.argwhere(np.isnan(yd)).flatten()) != 0:
                print(f"Warning! Video {self.dataFrame[0]['videoFilename'][np.argwhere(np.isnan(yd)).flatten()[0]]} contains NaN value for method {name}")
                continue
            # -- set color for box
            if metric == 'MAE' or metric == 'RMSE' or metric == 'TIME_REQUIREMENT' or metric == 'SNR':
                med = np.median(yd)
                col = str(min(200, 5 * int(med) + offset))
            if metric == 'CC' or metric == 'PCC' or metric == 'CCC':
                med = 1 - np.abs(np.median(yd))
                col = str(int(200 * med) + offset)

            # -- add box 
            fig.add_trace(go.Box(
                y=yd,
                name=name,
                boxpoints='all',
                jitter=.7,
                # whiskerwidth=0.2,
                fillcolor="rgba(" + col + "," + col + "," + col + ",0.5)",
                line_color="rgba(0,0,255,0.5)",
                marker_size=2,
                line_width=2)
            )

        gwidth = np.max(Y) / 10

        if title:
            tit = "Metric: " + metric
            top = 40
        else:
            tit = ''
            top = 10

        fig.update_layout(
            title=tit,
            yaxis_type=scale,
            xaxis_type="category",
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                # dtick=gwidth,
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

        # fig.show()
        return fig

    def saveStatsData(self, methods=None, metric='MAE', outfilename='statsData.csv'):
        """
        Saves statistics of data on disk 
        """
        Y = self.getStatsData(methods=methods, metric=metric, printTable=False)
        np.savetxt(outfilename, Y)

    def getStatsData(self, methods=None, metric='MAE', printTable=True):
        """
        Computes statistics of data 
        """
        # -- Method(s)
        if methods == None:
            methods = self.methods
        else:
            if set(methods) <= set(self.methods):
                raise ("Some method is wrong!")
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
        I = ss.iqr(Y, axis=0)
        M = np.median(Y, axis=0)
        Y = np.vstack((Y, M))
        Y = np.vstack((Y, I))

        methodNames = [x.upper() for x in self.methods]
        dataseNames = self.datasetNames
        dataseNames.append('Median')
        dataseNames.append('IQR')
        df = pd.DataFrame(Y, columns=methodNames, index=dataseNames)
        if printTable:
            display(df)

        return Y, df

    def __remove_outliers(self, df, factor=3.5):
        """
        Removes the outliers. A data point is considered an outlier if
        lies outside factor times the inter-quartile range of the data distribution 
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_out = df[~((df < (Q1 - factor * IQR)) | (df > (Q3 + factor * IQR))).any(axis=1)]
        return df_out

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

        if self.remove_outliers:
            res = pd.DataFrame(np.array(Y).T)
            res = self.__remove_outliers(res)
            res = res.to_numpy()
        else:
            res = np.array(Y).T

        return res

    def __getData(self):

        mag = self.mag
        metric = self.metric
        methods = self.methods

        # -- loop on datasets
        Y = []
        m_list = []
        for i, frame in enumerate(self.dataFrame):
            # -- loop on methods
            y = []
            for method in methods:
                vals = frame[frame['method'] == method][metric]
                if vals.empty:
                    continue
                m_list.append(method)
                if mag == 'min':
                    data = [v[np.argmin(v)] for v in vals]
                else:
                    data = [v[np.argmax(v)] for v in vals]
                y.append(data)
            y = np.array(y)

            if not self.join_data:
                Y.append(np.mean(y, axis=1))
            else:
                Y.append(y.T)

        if not self.join_data:
            res = np.array(Y)
        else:
            self.methods = m_list
            n_dpoints = [curr_y.shape[0] for curr_y in Y]
            if len(set(n_dpoints)) != 1:
                raise ("There should be the exact same number of elements in each dataset to join when 'join_data=True'")
            res = np.hstack(Y)

        if self.remove_outliers:
            res = pd.DataFrame(res)
            res = self.__remove_outliers(res)
            res = res.to_numpy()

        return res

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

        if not self.join_data:
            # -- method names intersection among datasets
            methods = set(mets[0])
            if N > 1:
                for m in range(1, N - 1):
                    methods.intersection(mets[m])
            methods = list(methods)
        else:
            methods = sum([list(m) for m in mets], [])
            if sorted(list(set(methods))) != sorted(methods):
                raise ("Found multiple methods with the same name... Please ensure using different names for each method when 'join_data=True'")

        self.methods = methods
        self.dataFrame = dataFrame
