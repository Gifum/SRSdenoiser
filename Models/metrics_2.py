# Code for custom metrics used in performance analysis of the SRSdenoiser neural network
### See the main and SI of the paper for additional details


import numpy as np
from scipy import signal
from skimage import metrics as sm




import tensorflow.keras.backend as K


def SNRcustom(GT,Y, axis=-1):
    '''
    Calculate the normalized MAE metric
    Input:
        GT: array of GT spectra of size (number of pixels in each spectrum, number of spectra)
        Y: array of Y spectra of size (number of pixels in each spectrum, number of spectra)
        axis: dimension of the input array over which the metric is computed
    Output:
        snr: array of size (number of pixels in each spectrum) containing the NMAE values for each spectrum 
    '''  
    
    y_true= (GT/np.max(GT,-1)[:,None])
    y_pred = (Y/np.max(Y,-1)[:,None])
    

    noise = np.sum(np.abs(y_pred - y_true),axis)
    signal = np.sum(np.abs(y_pred),axis)
    snr =  signal / noise

    return snr

def SNRcustom2(GT,Y, axis=-1):
    '''
    Calculate the normalized MSE metric
    Input:
        GT: array of GT spectra of size (number of pixels in each spectrum, number of spectra)
        Y: array of Y spectra of size (number of pixels in each spectrum, number of spectra)
        axis: dimension of the input array over which the metric is computed
    Output:
        snr: array of size (number of pixels in each spectrum) containing the NMSE values for each spectrum 
    '''  
    
    y_true= (GT/np.max(GT,-1)[:,None])
    y_pred = (Y/np.max(Y,-1)[:,None])
    
    snr =  np.mean(y_true,axis) / (np.sum((y_pred - y_true)**2,axis))

    return snr

def SNRcustom3(GT,Y,GT_freq_test, width=80,nbin=801):
    '''
    Custom SNR metrics defined in the paper
    Inputs:
        GT: array of GT spectra of size (number of pixels in each spectrum, number of spectra)
        Y: array of Y spectra of size (number of pixels in each spectrum, number of spectra)
        GT_freq_test: true spectral positions of the GT features (from metadata)
        width: range of frequencies around the peak that are considered to calculated the peak area (corresponds to epsilon in eq. 4 of the main text)
        nbin: length of the spectrum
    Outputs:
        SNR: custom SNR defined in eq. 4 of the main text of the paper
    '''  

    SNR=np.zeros(len(GT))
    for n in range(len(GT)):
        y_true= (GT[n]/np.max(GT[n]))
        y_pred = (Y[n]/np.max(Y[n]))
        GTfreq= GT_freq_test[n]

        index=np.array([])
        for p in GTfreq:
            wpre=np.max((p-width,0))
            wpost=np.min((p+width,len(y_true)))
            index = np.concatenate((index,np.arange(wpre,wpost)))

        index = (np.unique(index)).astype(int)
        index_n=[z for z in range(nbin) if not z in index]

        I= np.max(np.abs(y_pred[np.around(index)]))
 
        N_mean = np.mean(y_pred[np.around(index_n)])  
        N_std = np.std(y_pred[np.around(index_n)]) 


        SNR[n] = (I-N_mean) / (N_std)
    return SNR



def IsPeak(peaksGT,peaksY,tol=3.1,debug=0):
    ''' 
    Compare pairs of closest elements from two lists containing the spectral positions of the Raman modes of the GT and denoised spectra to assess if they are the same within a tunable value of tolerance (in pixels)
    Inputs:
        peaksGT: list of doubles (spectral positions of the features of the GT spectrum)
        peaksY: list of doubles (spectral positions of the features of the denoised spectrum)
        tol: set the tolerance in pixels within which two numbers are considered the same spectral position
        debug: set to 1 to print the pair of numbers that have been considered
    Outputs:
        output: elements of peaksGT for which a corresponding element has been found in peaksY
    '''    
        
    
    peaksGT = np.array(peaksGT)
    peaksY = np.array(peaksY)
    output = np.array([],np.dtype(np.int32))


    if len(peaksGT)>len(peaksY):
        peaksA=peaksGT
        peaksB=peaksY
    else:
        peaksA=peaksY
        peaksB=peaksGT

    closeA = np.zeros((len(peaksB),2),dtype='int32')

    for i in range(len(peaksB)):

        lo=np.argwhere(peaksA<peaksB[i])
        closeA[i,0] = lo[-1].item() if len(lo)>0 and (np.abs(peaksB[i]-peaksA[lo[-1].item()])<tol) else -1
        up=np.argwhere(peaksA>=peaksB[i])
        closeA[i,1] = up[0].item() if len(up)>0 and (np.abs(peaksB[i]-peaksA[up[0].item()])<tol)  else -1
        
    list_rem=[]
    
    for i in range(len(peaksB)):
        if closeA[i,0] not in list_rem and closeA[i,0]>=0:
            list_rem.append(closeA[i,0])
            if debug:
                print(peaksB[i], peaksA[closeA[i,0]])
        elif closeA[i,1] not in list_rem and closeA[i,1]>=0:
            list_rem.append(closeA[i,1])
            if debug:
                print(peaksB[i], peaksA[closeA[i,1]])

    output=peaksA[list_rem]
            

     
    return output





def PeakFinder(gt, y, noisy=[], debug=0, onlyPeaks=False):
    '''
    Find positive and negative edges in a noisy spectrum with a tolerance on noise and prominence ruled by a corresponding GT spectrum 
    Inputs:
        gt: GT spectrum
        y: noisy spectrum without the baseline
        noisy: raw spectrum (with baseline) associated to y (only used if debug=1)
        debug: set to 1 to plot gt,y and noisy with the found edges
        onlyPeaks: set to True to find only positive peaks instead of edges 
    Outputs:
        peaksGT: list of edges of the GT spectrum
        peaksY: list of edges of the noisy spectrum
    '''     
 

    ## GT
    gt = gt /np.max(gt)

    ### Peaks using derivative
    epsilon_gt = np.mean(np.abs(gt))/2
    DD = np.diff(gt)
    DD2=np.diff(DD)
    DD = np.pad(DD,(1,0),'reflect')
    DD2 = np.pad(DD2,(2,0),'reflect')
    
    peaksGT = np.where(np.diff(np.sign(DD)))[0]
    
    
   
    if onlyPeaks:
        #onlyPeaks Turn to True to detect peaks and not edges
        del peaksGT
        
        peaksXzero = np.where(np.diff(np.sign(DD)))[0]
    
        #Divide the indexes before and after the zeroes of the diff
        peaksBef = peaksXzero[::2]
        peaksAft = peaksXzero[1::2]

        if len(peaksAft)<len(peaksBef):
            peaksAft=np.append(peaksAft,peaksBef[-1])

        peaksGT=np.zeros((len(peaksBef),), dtype=int)

        for ii in range(len(peaksAft)):
            if np.abs(gt)[peaksBef[ii]] > np.abs(gt)[peaksAft[ii]]:
                peaksGT[ii] = peaksBef[ii]
            else:
                peaksGT[ii] = peaksAft[ii]
                

    ### Check on curvature

    DD2Condition = np.sign(DD2[peaksGT+1])*np.sign(gt[peaksGT]) < 0 


    peaksGT = [peaksGT[p] for p in range(len(peaksGT)) if (np.abs(gt)[peaksGT[p]]>epsilon_gt) and DD2Condition[p]]    
    peaksGT = np.array(peaksGT)

    AmpGT=np.min(np.abs(gt[peaksGT]))


    ## Res
    y=y/np.max(y)

    epsilon = 0.5 * AmpGT
    
    ### Peaks using derivative

    DD = np.diff(y)
    DD = np.pad(DD,(1,0),'reflect')

    DD2=np.diff(DD)
    DD2 = np.pad(DD2,(1,0),'reflect')

    peaksY = np.where(np.diff(np.sign(DD)))[0]
    
   
    if onlyPeaks:
        #onlyPeaks Turn to True to detect peaks and not edges
        del peaksY
        peaksXzero = np.where(np.diff(np.sign(DD)))[0]

        #Divide the indexes before and after the zeroes of the diff
        peaksBef = peaksXzero[::2]
        peaksAft = peaksXzero[1::2]

        if len(peaksAft)<len(peaksBef):
            peaksAft=np.append(peaksAft,peaksBef[-1])

        peaksY=np.zeros((len(peaksBef),), dtype=int)

        for ii in range(len(peaksAft)):
            if np.abs(y)[peaksBef[ii]] > np.abs(y)[peaksAft[ii]]:
                peaksY[ii] = peaksBef[ii]
            else:
                peaksY[ii] = peaksAft[ii]

                

    ### Check on curvature

    DD2Condition = np.sign(DD2[peaksY+1])*np.sign(y[peaksY]) < 0 

    peaksY = [peaksY[p] for p in range(len(peaksY)) if (np.abs(y[peaksY[p]])>epsilon) and DD2Condition[p]]    


    ### Refining
    prominences = signal.peak_prominences(np.abs(y), peaksY, wlen=100)[0]
    epsilonProm = 5e-3
    peaksY = [peaksY[p] for p in range(len(peaksY)) if prominences[p]>epsilonProm ]    



    prominences = signal.peak_prominences(np.abs(y), peaksY, wlen=250)[0]

    idx_sort = np.argsort(prominences)
    idx_sort = idx_sort[::-1]

    peaksY=np.array(peaksY)
    peaksY = peaksY[idx_sort]

    peaksY_sort = peaksY[0:len(peaksGT)+2]

    peaksY_temp=np.copy(peaksY)
    del(peaksY)
    peaksY=peaksY_sort

    if len(peaksY)>len(peaksGT):
        prominences = signal.peak_prominences(np.abs(y), peaksY, wlen=400)[0]
        epsilonProm = prominences[len(peaksGT)-1]*0.8
        peaksY = [peaksY[p] for p in range(len(peaksY)) if prominences[p]>epsilonProm ]

        # Refining with averages

        ww = signal.peak_widths(np.abs(y), peaksY, rel_height=0.5)[0]#, rel_height=0.5)
        ww = ww.astype(int)
        epsilonMean=epsilon*.3

        ww1 = np.zeros(len(ww))+2
        ss =np.zeros([len(ww),2])
        ss[:,1] = peaksY-ww
        www1 = [np.max(ss,1).astype(int)]
        idx_ww= np.argwhere(np.sign(y[peaksY])==np.sign(y[tuple(www1)]))
        ww1[idx_ww]=ww[idx_ww]
        ww1=ww1.astype(int)

        ww2 = np.zeros(len(ww))+2
        ss =np.full([len(ww),2],len(y)-1)
        ss[:,1] = peaksY +ww
        www2 = [np.min(ss,1).astype(int)]
        idx_ww= np.argwhere(np.sign(y[peaksY])==np.sign(y[tuple(www2)]))
        ww2[idx_ww]=ww[idx_ww]
        ww2=ww2.astype(int)



        peaksY = [peaksY[p] for p in range(len(peaksY)) if (np.abs(y[peaksY[p]])-np.abs(np.mean(y[peaksY[p]-ww1[p]:peaksY[p]+ww2[p]]))>epsilonMean) ]



    peaksY=np.unique(peaksY)
    
    if debug:
        
        if len(peaksY)==0:
            peaksY_plot=[0]
        else:
            peaksY_plot=peaksY
        
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        # Plots
        fig = make_subplots(rows=1, cols=4, horizontal_spacing = 0.0)



        fig.append_trace(go.Scatter(x=np.arange(0,800), y=DD,
            mode='lines', #line=dict(color="#0000ff"),
            name="D1",
                                 #line=dict(color='blue'),
                                 showlegend=True
        ), row=1, col=1)

        fig.append_trace(go.Scatter(x=np.arange(0,800), y=(y),
        mode='lines', #line=dict(color="#0000ff"),
        name="Retrieved",
                                 #line=dict(color='blue'),
                                 showlegend=True
        ), row=1, col=2)



        fig.append_trace(go.Scatter(x=peaksY_plot, y=y[peaksY_plot],
            mode='markers', #line=dict(color="#0000ff"),
            name="Points",
                                 line=dict(color='blue'),
                                 showlegend=True
        ), row=1, col=2)

       

        fig.append_trace(go.Scatter(x=peaksY_plot, y=DD[peaksY_plot],
            mode='markers', #line=dict(color="#0000ff"),
            name="Points",
                                 line=dict(color='red'),
                                 showlegend=True
        ), row=1, col=1)


        fig.append_trace(go.Scatter(x=peaksGT, y=gt[peaksGT],
            mode='markers', #line=dict(color="#0000ff"),
            name="Points",
                                 line=dict(color='blue'),
                                 showlegend=True
        ), row=1, col=3)

        fig.add_hline(y=epsilon_gt, row=1, col=3)
        fig.add_hline(y=-epsilon_gt, row=1, col=3)

        fig.add_hline(y=np.mean(y[0:100]), row=1, col=2)


        fig.append_trace(go.Scatter(x=np.arange(0,800), y=gt,
            mode='lines', #line=dict(color="#0000ff"),
            name="GT",
                                 line=dict(color='green'),
                                 showlegend=True
        ), row=1, col=3)

        fig.append_trace(go.Scatter(x=np.arange(0,800), y=noisy,
            mode='lines', #line=dict(color="#0000ff"),
            name="Noisy",
                                 line=dict(color='black'),
                                 showlegend=True
        ), row=1, col=4)


        fig.show()
        print(peaksGT, len(peaksGT),'\n',peaksY, len(peaksY))
        
    return peaksGT,peaksY


def PeakMetrics(Noisy,GT,Y, GT_freq_test, tole=1.1):

    '''
    Calculate classification metrics using the output of PeakFinder and IsPeak 
    Inputs:
        Noisy: array of noisy spectra (only used for the PeakFinder method in debugging mode)
        GT: array of GT spectra
        Y: array of denoised spectra whose performance needs to be evaluated
        GT_freq_test: true spectral positions of the GT features (from metadata)
        tole: set the tolerance for the IsPeak method
    Outputs:
        precision= TP/(TP+FP)
        recall= TP/(TP+FN)
        F1= 2* precision*recall/(precision+recall)
        TP: true positives, i.e. number of features in Y that correspond to a feature in GT 
        FP: false positives, i.e. number of features in Y that do not correspond to a feature in GT
        FN: false negative, i.e. number of features in GT that do not have a corresponding one in Y
        TP_modes: TP evaluated using GT_freq_test as ground truth for the spectral position of GT
    '''    
    
    length_samples = len(Noisy[:,0])
    TP = np.array([],np.dtype(np.int32))
    TP_modes = np.array([],np.dtype(np.int32))
    FP = np.array([],np.dtype(np.int32))
    FN = np.array([],np.dtype(np.int32))
    precision = np.array([])
    recall = np.array([])
    F1 = np.array([])

    for nn in range(length_samples):

        noisy=Noisy[nn]
        gt=GT[nn]
        y=Y[nn]

        peaksGT,peaksY = PeakFinder(gt,y)


        founds = IsPeak(peaksGT,peaksY,tol=tole)

        tp = len(founds)
        fp = len(peaksY)-len(founds)
        if fp<0:
            print('Error: fp is negative at sample '+str(nn))
        fn = len(peaksGT)-len(founds)
        if fn<0:
            print('Error: fn is negative at sample '+str(nn))

        if (tp+fp==0):
            prec=0
            #rec=-1.5
            print('Error: tp+fp is zero at sample '+str(nn))
        else:  
            prec = tp / (tp+fp)
            #rec= tp / (tp+fn)

        if (tp+fn==0):
            #prec=-1.5
            rec=0
            print('Error: tp+fn is zero at sample '+str(nn))
        else:  
            #prec = tp / (tp+fp)
            rec= tp / (tp+fn)



        TP = np.append(TP,tp)
        FP = np.append(FP,fp)
        FN = np.append(FN,fn)

        precision = np.append(precision, prec)
        recall =  np.append(recall,rec)
        if prec+rec ==0:
            f1 = 0.
        else:
            f1 = 2 * (prec*rec/(prec+rec))

        F1 = np.append(F1,f1)

        peaksGT_modes = GT_freq_test[nn].reshape(len(GT_freq_test[nn]),)
        founds_modes = IsPeak(peaksGT_modes,peaksY,tol=tole)
        tp_modes = len(founds_modes)/len(peaksGT_modes)
        TP_modes = np.append(TP_modes,tp_modes)


    
    return precision, recall, F1, TP, FP, FN, TP_modes



def compute_metrics2(Y,Noisy,GT,GT_freq_test,nbin=801):
    '''
    For an array of denoised spectra, calculate all the metrics used in the paper and store them in a dictionary
    '''    

    precision, recall, F1, TP, FP, FN, TP_modes = PeakMetrics(Noisy,GT,Y,GT_freq_test, tole=1.1)
    sensitivity = TP / (TP+FN)
    
    snr=SNRcustom(GT,Y, axis=-1)
    csnr=SNRcustom3(GT,Y,GT_freq_test, width=80,nbin=nbin)

    test_len = np.shape(Noisy)[0]
    ssim_list=np.zeros((test_len,))
    nmse_list=np.zeros((test_len,))
    csnr_list=np.zeros((test_len,))

    for nn in range(0,test_len):
        gt = GT[nn]
        gt = np.float32(gt)
        y = Y[nn]


        gt = gt / np.max(gt)
        y = y / np.max(y)


        DR = 1
        ssim, ssim_map = sm.structural_similarity(gt, y, win_size=27, gradient=False,data_range=DR,
        channel_axis=None, multichannel=False, gaussian_weights=False, full=True, use_sample_covariance=False, K1=0.001,K2=0.003,sigma=1.5)

        nmse = sm.normalized_root_mse(gt, y, normalization='min-max')


        ssim_list[nn]=ssim
        nmse_list[nn]=nmse

        snr[nn]/=len(GT_freq[4000+nn])

    
    metrics={"ssim":ssim_list,
             "nmse":nmse_list,
             "csnr":csnr,
             "nmae":1/snr, 
             "precision":precision, 
             "sensitivity": sensitivity,
                 "recall":recall, 
                 "F1":F1,
                 "TP":TP, 
                  "FP":FP,
                 "FN":FN,
                 "TP_modes":TP_modes
        
    }
    
    return metrics