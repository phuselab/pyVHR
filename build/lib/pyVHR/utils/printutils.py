import plotly.graph_objects as go
import numpy as np

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
        
def multiplot(x=None, y=None, name=None, zeroMean=True, title="Signal", height=400, width=800):

    fig = go.Figure()

    if not np.any(y):
        return
    else:
        if y.ndim == 1:
            c = 1
            n = y.shape[0]
            if zeroMean:
                z = y-y.mean()
            if not np.any(x):
                x = np.linspace(0,n-1,n)
            fig.add_trace(go.Scatter(x=x,y=z,name=name))
        else:
            c,n = y.shape
            if not np.any(x):
                x = np.linspace(0,n-1,n)
            for i in range(c):
                z = y[i] 
                if name:
                    s = name[i]
                else:
                    s = "sig" + str(i)
                if zeroMean:
                    z = y[i]-y[i].mean()
                    
                fig.add_trace(go.Scatter(x=x,y=z,name=s))
    
    fig.update_layout(height=height, width=width, title=title,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#7f7f7f")
    )

    fig.show()
    
    return fig