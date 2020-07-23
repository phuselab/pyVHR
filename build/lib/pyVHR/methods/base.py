import numpy as np
import ast
import plotly.graph_objects as go
from scipy.signal import medfilt, detrend
from abc import ABCMeta, abstractmethod
from importlib import import_module
from ..signals.bvp import BVPsignal
from ..utils import filters, printutils
from ..utils import detrending

def methodFactory(methodName, *args, **kwargs):
    try:
        moduleName = methodName.lower()
        className = methodName.upper()
        methodModule = import_module('.methods.' + moduleName, package='pyVHR')
        classOBJ = getattr(methodModule, className)
        obj = classOBJ(**kwargs)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of pyVHR method collection!'.format(methodName))

    return obj
    
class VHRMethod(metaclass=ABCMeta):
    """
    Manage VHR approaches (parent class for new approach)
    """
    
    def __init__(self, **kwargs):
        self.video = kwargs['video']
        self.verb = kwargs['verb']
        
    @abstractmethod
    def apply(self, X):
        pass

    def runOffline(self, **kwargs):
        
        # -- parse params
        startTime, endTime, winSize, timeStep, zeroMeanSTDnorm, BPfilter, minHz, maxHz, detrFilter, \
        detrMethod, detrLambda = self.__readparams(**kwargs)
        
        fs = self.video.frameRate
        
        # -- check times
        if endTime > self.video.duration:
            endTime = self.video.duration
        assert startTime <= endTime, "Time interval error!"
        assert timeStep > 0, "Time step must be positive!"
        assert winSize < (endTime-startTime),"Winsize too big!"

        # -- verbose prints
        if '1' in str(self.verb):
            self.__verbose(startTime, endTime, winSize)
        
        if self.video.doEVM is True:
            self.video.applyEVM()
        else:
            self.video.processedFaces = self.video.faces

        timeSteps = np.arange(startTime,endTime,timeStep)
        T = startTime    # times where bpm are estimated
        RADIUS = winSize/2

        bpmES = []     # bmp estimtes
        timesES = []   # times of bmp estimtes

        # -- loop on video signal chunks
        startFrame = int(T*self.video.frameRate)
        count = 0
        while T <= endTime:
            endFrame = np.min([self.video.numFrames, int((T+RADIUS)*self.video.frameRate)])

            # -- extract ROIs on the frame range
            self.frameSubset = np.arange(startFrame, endFrame)

            self.ROImask = kwargs['ROImask']

            # -- type of signal extractor
            if self.ROImask == 'rect':
                rects = ast.literal_eval(kwargs['rectCoords'])
                self.rectCoords = []
                for x in rects:
                    rect = []
                    for y in x:
                        rect.append(int(y))
                    self.rectCoords.append(rect)
                self.video.setMask(self.ROImask, rectCoords=self.rectCoords)
            elif self.ROImask == 'skin_adapt':
                self.video.setMask(self.ROImask, skinThresh_adapt=float(kwargs['skinAdapt']))
            elif self.ROImask == 'skin_fix':
                threshs = ast.literal_eval(kwargs['skinFix'])
                self.threshSkinFix = [int(x) for x in threshs]
                self.video.setMask(self.ROImask, skinThresh_fix=self.threshSkinFix)
            else:
                raise ValueError(self.ROImask + " : Unimplemented Signal Extractor!")
                
            self.video.extractSignal(self.frameSubset, count)

            # -- RGB computation  
            RGBsig = self.video.getMeanRGB()
            
            # -- print RGB raw data
            if '2' in str(self.verb):
                printutils.multiplot(y=RGBsig, name=['ch B', 'ch R','ch G'], title='RGB raw data')
                        
            # -- RGBsig preprocessing
            if zeroMeanSTDnorm:
                RGBsig = filters.zeroMeanSTDnorm(RGBsig)      
            if detrFilter:
                if detrMethod == 'tarvainen':
                    #TODO controllare il detrending di tarvainen
                    RGBsig[0] = detrending.detrend(RGBsig[0], detrLambda)
                    RGBsig[1] = detrending.detrend(RGBsig[1], detrLambda)
                    RGBsig[2] = detrending.detrend(RGBsig[2], detrLambda)
                else:
                    RGBsig = detrend(RGBsig)
            if BPfilter:
                RGBsig = filters.BPfilter(RGBsig, minHz, maxHz, fs)
            
            # -- print postproce
            if '2' in str(self.verb):
                printutils.multiplot(y=RGBsig, name=['ch B', 'ch R','ch G'], title='RGB postprocessing')
            
            # -- apply the selected method to estimate BVP
            rPPG = self.apply(RGBsig)
                        
            # BVP postprocessing 
            startTime = np.max([0, T-winSize/self.video.frameRate])
            bvpChunk = BVPsignal(rPPG, self.video.frameRate, startTime, minHz, maxHz, self.verb)
            
            # -- post processing: filtering 
            
            # TODO: valutare se mantenere!!
            #bvpChunk.data = filters.BPfilter(bvpChunk.data, bvpChunk.minHz, bvpChunk.maxHz, bvpChunk.fs)

            if '2' in str(self.verb):
                bvpChunk.plot(title='BVP estimate by ' + self.methodName)
          
            # -- estimate BPM by PSD
            bvpChunk.PSD2BPM(chooseBest=True)

            # -- save the estimate
            bpmES.append(bvpChunk.bpm)
            timesES.append(T)

            # -- define the frame range for each time step            
            T += timeStep
            startFrame = np.max([0, int((T-RADIUS)*self.video.frameRate)])

            count += 1

        # set final values
        self.bpm = np.array(bpmES).T
        
        # TODO controllare se mettere o no il filtro seguente
        #self.bpm = self.bpm_time_filter(self.bpm, 3)
        self.times = np.array(timesES)
        
        return self.bpm, self.times

    @staticmethod    
    def makeMethodObject(video, methodName='ICA'):
        if methodName == 'CHROM':
            m = methods.CHROM(video)
        elif methodName == 'LGI':
            m = methods.LGI(video)
        elif methodName == 'SSR':
            m = methods.SSR(video)
        elif methodName == 'PBV':
            m = methods.PBV(video)
        elif methodName == 'POS':
            m = methods.POS(video)
        elif methodName == 'Green':
            m = methods.Green(video)
        elif methodName == 'PCA':
            m = methods.PCA(video)
        elif methodName == 'ICA':
            m = methods.ICA(video)
        else:
            raise ValueError("Unknown method!")
        return m

    def __readparams(self, **kwargs):
        
        # get params from kwargs or set default
        if 'startTime' in kwargs:
            startTime = float(kwargs['startTime'])
        else:
            startTime = 0
        if 'endTime' in kwargs:
            if kwargs['endTime']=='INF':
                endTime = np.Inf
            else:
                endTime = float(kwargs['endTime'])
        else:
            endTime=np.Inf
        if 'winSize' in kwargs:
            winSize = int(kwargs['winSize'])
        else:
            winSize = 5
        if 'timeStep' in kwargs:
            timeStep = float(kwargs['timeStep'])
        else:
            timeStep = 1
        if 'zeroMeanSTDnorm' in kwargs:
            zeroMeanSTDnorm = int(kwargs['zeroMeanSTDnorm'])
        else:
            zeroMeanSTDnorm = 0
        if 'BPfilter' in kwargs:
            BPfilter = int(kwargs['BPfilter'])
        else:
            BPfilter = 1
        if 'minHz' in kwargs:
            minHz = float(kwargs['minHz'])
        else:
            minHz = .75
        if 'maxHz' in kwargs:
            maxHz = float(kwargs['maxHz'])
        else:
            maxHz = 4.
        if 'detrending' in kwargs:
            detrending = int(kwargs['detrending'])
        else:
            detrending = 0
        if detrending:
            if 'detrLambda' in kwargs:
                detrLambda = kwargs['detrLambda']
            else:
                detrLambda = 10
        else:
            detrLambda = 10
        if 'detrMethod' in kwargs:
            detrMethod = kwargs['detrMethod']
        else:
            detrMethod = 'tarvainen'
            
        return startTime, endTime, winSize, timeStep, zeroMeanSTDnorm, BPfilter, minHz, maxHz,\
                detrending, detrMethod, detrLambda
    
    def RMSEerror(self, bvpGT):
        """ RMSE: """

        diff = self.__diff(bvpGT)
        n,m = diff.shape  # n = num channels, m = bpm length
        df = np.zeros(n)
        for j in range(m):
            for c in range(n):
                df[c] += np.power(diff[c,j],2)

        # -- final RMSE
        RMSE = np.sqrt(df/m)
        return RMSE

    def MAEerror(self, bvpGT):
        """ MAE: """

        diff = self.__diff(bvpGT)
        n,m = diff.shape  # n = num channels, m = bpm length
        df = np.sum(np.abs(diff),axis=1)

        # -- final MAE
        MAE = df/m
        return MAE

    def MAXError(self, bvpGT):
        """ MAE: """

        diff = self.__diff(bvpGT)
        n,m = diff.shape  # n = num channels, m = bpm length
        df = np.max(np.abs(diff),axis=1)

        # -- final MAE
        MAX = df
        return MAX

    def PearsonCorr(self, bvpGT):
        from scipy import stats

        diff = self.__diff(bvpGT)
        bpmES = self.bpm
        n,m = diff.shape  # n = num channels, m = bpm length
        CC = np.zeros(n)
        for c in range(n):
            # -- corr
            r,p = stats.pearsonr(diff[c,:]+bpmES[c,:],bpmES[c,:])
            CC[c] = r
        return CC

    def printErrors(self, bvpGT):
        RMSE = self.RMSEerror(bvpGT)
        MAE = self.MAEerror(bvpGT)
        CC = self.PearsonCorr(bvpGT)
        print('\nErrors:')
        print('        RMSE: ' + str(RMSE))
        print('        MAE : ' + str(MAE))
        print('        CC  : ' + str(CC))

    def displayError(self, bvpGT):
        bpmGT = bvpGT.bpm
        timesGT = bvpGT.times
        bpmES = self.bpm
        timesES = self.times
        diff = self.__diff(bvpGT)
        n,m = diff.shape  # n = num channels, m = bpm length
        df = np.abs(diff)
        dfMean = np.around(np.mean(df,axis=1),1)

        # -- plot errors
        fig = go.Figure()
        name = 'Ch 1 (µ = ' + str(dfMean[0])+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=df[0,:], name=name, mode='lines+markers'))
        if n > 1:
            name = 'Ch 2 (µ = ' + str(dfMean[1])+ ' )'
            fig.add_trace(go.Scatter(x=timesES, y=df[1,:], name=name, mode='lines+markers'))
            name = 'Ch 3 (µ = ' + str(dfMean[2])+ ' )'
            fig.add_trace(go.Scatter(x=timesES, y=df[2,:], name=name, mode='lines+markers'))
        fig.update_layout(xaxis_title='Times (sec)', yaxis_title='MAE', showlegend=True)
        fig.show()
        
        # -- plot bpm Gt and ES 
        fig = go.Figure()
        GTmean = np.around(np.mean(bpmGT),1)
        name = 'GT (µ = ' + str(GTmean)+ ' )'
        fig.add_trace(go.Scatter(x=timesGT, y=bpmGT, name=name, mode='lines+markers'))
        ESmean = np.around(np.mean(bpmES[0,:]),1)
        name = 'ES1 (µ = ' + str(ESmean)+ ' )'
        fig.add_trace(go.Scatter(x=timesES, y=bpmES[0,:], name=name, mode='lines+markers'))
        if n > 1:
            ESmean = np.around(np.mean(bpmES[1,:]),1)
            name = 'ES2 (µ = ' + str(ESmean)+ ' )'
            fig.add_trace(go.Scatter(x=timesES, y=bpmES[1,:], name=name, mode='lines+markers'))
            ESmean = np.around(np.mean(bpmES[2,:]),1)
            name = 'E3 (µ = ' + str(ESmean)+ ' )'
            fig.add_trace(go.Scatter(x=timesES, y=bpmES[2,:], name=name, mode='lines+markers'))
        
        
        
        fig.update_layout(xaxis_title='Times (sec)', yaxis_title='BPM', showlegend=True)
        fig.show()



    def __diff(self, bvpGT):
        bpmGT = bvpGT.bpm
        timesGT = bvpGT.times
        bpmES = self.bpm
        timesES = self.times
        n,m = bpmES.shape  # n = num channels, m = bpm length

        diff = np.zeros((n,m))
        for j in range(m):
            t = timesES[j]
            i = np.argmin(np.abs(t-timesGT))
            for c in range(n):
                diff[c,j] = bpmGT[i]-bpmES[c,j]
        return diff

    def bpm_time_filter(self, bpm, w_len):

        n_sig = bpm.shape[0]
        filtered_bpm = []
    
        for s in range(n_sig):
            x = bpm[s,:]
            x = medfilt(x, w_len)
            filtered_bpm.append(x)

        filtered_bpm = np.vstack(filtered_bpm)

        return filtered_bpm
    
        
    def __verbose(self, startTime, endTime, winSize):
        print("\n    * %s params: start time = %.1f, end time = %.1f, winsize = %.1f (sec)" 
              %(self.methodName, startTime, endTime, winSize))
        