import os
import sys
import time

from pyVHR.analysis.multi_method_suite import MultiMethodSuite, TestResult
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.analysis.stats import StatAnalysis
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def main():

    t1 = time.time()

    # Evaluation
    evaluation = Pipeline()
    results = evaluation.run_on_dataset(configFilename="pyVHR/analysis/ubfc1_evaluation.cfg", verb=True)
    results.saveResults("results/ufc1_quick_evaluation.h5")
    print(" #### Time consumed in the evaluation: {} seconds...".format(time.time() - t1))

    # Visulization of results
    st = StatAnalysis(filepath="results/ufc1_quick_evaluation.h5")
    y_df, fig_stats = st.run_stats()
    fig = st.displayBoxPlot(metric='MAE')
    fig.show()
    fig = st.displayBoxPlot(metric='RMSE')
    fig.show()
    fig = st.displayBoxPlot(metric='SNR')
    fig.show()


if __name__ == '__main__':
    main()
