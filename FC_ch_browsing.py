# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:41:39 2023

@author: smontero
"""

# Import common libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

#from itertools import compress
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import pathlib
import numpy as np
import numpy.matlib
import os.path as op
import seaborn as sns
from scipy import stats, signal
from sklearn.svm import SVR


# Import MNE processing
import mne
from mne.viz import plot_compare_evokeds
from mne import Epochs, events_from_annotations, set_log_level, pick_types

# Import MNE-NIRS processing
import mne_nirs
import mne_connectivity
from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.channels import picks_pair_to_idx
from mne.preprocessing.nirs import beer_lambert_law, optical_density,\
    temporal_derivative_distribution_repair, scalp_coupling_index
#from mne_nirs.signal_enhancement import enhance_negative_correlation


# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#import matplotlib.animation as animation
#from sklearn.decomposition import NMF
from sklearn.metrics import r2_score
import matplotlib.colors as colors

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix

# Ours
from SQE_plot3Dto2D_function import SQE_3Dto2D
from pysnirf2 import Snirf
# %%
def get_2D_coords(raw_intensity):
    fileName = raw_intensity.filenames[0]
    snirfObj = Snirf(fileName,'r+')
    midpX, midpY = SQE_3Dto2D(snirfObj)
    midpX = np.concatenate((midpX,midpX))
    midpY = np.concatenate((midpY,midpY))

    ch_names = raw_intensity.ch_names
    WLs = snirfObj.nirs[0].probe.wavelengths
    flag_sort = 0
    for i, ch in enumerate(snirfObj.nirs[0].data[0].measurementList):
        snirfObj_str = 'S'+str(ch.sourceIndex)+'_'+'D'+str(ch.detectorIndex)+' '+str(WLs[ch.wavelengthIndex-1])
        if snirfObj_str!=ch_names[i] and snirfObj_str!=ch_names[i]+'.0':
            print('Alert, %s and %s ML does not match'%(snirfObj_str,ch_names[i]))
            flag_sort = 1 
        #else:
            #    print('%s == %s'%(snirfObj_str,ch_names[i]))
    
    if flag_sort==1:
        print('We need to reorder the ML before changing the 2D coords.')
        
    if flag_sort==0:
        for i,ch in enumerate(raw_intensity.info['chs']):
            ch['loc'][:3] = [midpX[i],midpY[i],0]
            
#%%
def seedcorr_map_surf(raw_haemo,sub,roi,roi_label,val = 'r',thres=0, sqw = 0):
    
    n_ch_roi = len(roi)
    raw_haemo_hbr = raw_haemo.copy()
    raw_haemo_hbo = raw_haemo.copy()
    raw_haemo_hbo.pick_types(fnirs='hbo')
    raw_haemo_hbr.pick_types(fnirs='hbr')
    ch_names_hbo_mne = np.array(raw_haemo_hbo.ch_names)
    ch_names_hbr_mne = np.array(raw_haemo_hbr.ch_names)
    n_channels = len(ch_names_hbo_mne)
    subj = raw_haemo_hbr.info['subject_info']['his_id']
    
    cm = mpl.colormaps['bwr']
    cm_ = cm(np.linspace(0,1,100))
    cm_[49-int((thres*100)/2) : 49+int((thres*100)/2),:] = np.array([1.0, 1.0, 1.0, 1.0])
    mycm = mpl.colors.ListedColormap(cm_)
    tansf_sp = (-0.0025,-0.002,0,0.097)
    extrapol = 'local'
    mask = np.zeros((n_channels),dtype=bool)
    mask_par = dict(marker='o', markerfacecolor='greenyellow', markeredgecolor='greenyellow',
                    linewidth=0, markersize=10)
    fig = plt.figure(layout="constrained",figsize=(16,9))
    if sqw >0:
        gs = mpl.gridspec.GridSpec(3, n_ch_roi+1,figure=fig)  
    else:
        gs = mpl.gridspec.GridSpec(3, n_ch_roi,figure=fig)  
    
    # #hbo
    # am_fname = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory\results\sub-'+sub +'_RS_run-'+sess+'_gsr_hbo_am.csv')
    # csv_nt = pd.read_csv(am_fname)
    # am_nt_hbo = csv_nt.values
    # np.fill_diagonal(am_nt_hbo,0)
    # pval_fname = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory\results\sub-'+sub +'_RS_run-'+sess+'_gsr_hbo_pval.csv')
    # csv_nt_pval = pd.read_csv(pval_fname)
    # pval_nt_hbo = csv_nt_pval.values
    # np.fill_diagonal(pval_nt_hbo,1)
    # am_nt_hbo[np.where(pval_nt_hbo>=0.01)] = 0
    # if val=='z':
    #     #am_nt_hbo = np.arctanh(am_nt_hbo)
    #     #Z = abs(.5*log((1+obj.R)./(1-obj.R))).*sign(obj.R);
    #     #Z(Z>6)=6;  % Fix the R=1 -> inf;  tanh(6) ~1 so cut there to keep the scale
    #     #Z(Z<-6)=-6;
    #     am_nt_hbo = abs(0.5 * np.log((1+am_nt_hbo)/(1-am_nt_hbo))  )*np.sign(am_nt_hbo)
    #     am_nt_hbo[np.where(am_nt_hbo>6)] = 6
    #     am_nt_hbo[np.where(am_nt_hbo<-6)] = -6
    # thres_hbo = np.max(abs(am_nt_hbo))*thres
    # am_nt_hbo[np.where(abs(am_nt_hbo)<thres_hbo)] = 0    
    # ch_names_hbo_nt = csv_nt.columns.to_numpy()
    
    #hbr sessions 1 and 2
    A = np.zeros((n_channels,n_ch_roi,2))
    for sess in range(2):
        print('Session:'+str(sess))
        am_fname = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory\results\sub-'+sub +'_RS_run-'+str(sess+1)+'_gsr_hbr_am.csv')
        csv_nt = pd.read_csv(am_fname)
        am_nt_hbr = csv_nt.values
        np.fill_diagonal(am_nt_hbr,0)
        
        sq_fname = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory\results\sub-'+sub +'_RS_run-'+str(sess+1)+'_gsr_hbx_sqmat.csv')
        sq_csv_nt = pd.read_csv(sq_fname)
        sq_nt_hbx = sq_csv_nt.values
        
        if sqw == 1:
          am_nt_hbr = (am_nt_hbr*sq_nt_hbx)*(sq_nt_hbx.T)
        
        # pval_fname = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory\results\sub-'+sub +'_RS_run-'+str(sess+1)+'_gsr_hbr_pval.csv')
        # csv_nt_pval = pd.read_csv(pval_fname)
        # pval_nt_hbr = csv_nt_pval.values
        # np.fill_diagonal(pval_nt_hbr,1)
        # am_nt_hbr[np.where(pval_nt_hbr>=0.01)] = 0
        if val=='z':
            #am_nt_hbr = np.arctanh(am_nt_hbr)
            am_nt_hbr = abs(0.5 * np.log((1+am_nt_hbr)/(1-am_nt_hbr))  )*np.sign(am_nt_hbr)
            am_nt_hbr[np.where(am_nt_hbr>6)] = 6
            am_nt_hbr[np.where(am_nt_hbr<-6)] = -6
        
        thres_hbr = np.max(abs(am_nt_hbr))*thres
        am_nt_hbr_to_A = am_nt_hbr.copy()
        am_nt_hbr[np.where(abs(am_nt_hbr)<thres_hbr)] = 0
        ch_names_hbr_nt = csv_nt.columns.to_numpy()
        
        if np.all(ch_names_hbr_mne == ch_names_hbr_nt):
            print('ML matches')
        else:
            print('ML does not match!')
            #return -1
    
        j = 0
        for ich in roi:
            
            seed = picks_pair_to_idx(raw_haemo_hbr,[ich])[0]
            A[:,j,sess] = am_nt_hbr_to_A[:,seed]
            mask[seed] = True
            # if val=='z':
            #     vmin = -np.max(abs(am_nt_hbo))#[:,seed]))
            #     vmax = np.max(abs(am_nt_hbo))#[:,seed]))
            #     cb_label = "Fisher's Z"
            # else:
            #     vmin = -1
            #     vmax = 1
            #     cb_label = "Pearson's r"
    
            # ax = fig.add_subplot(gs[0,j])
            # im,cn = mne.viz.plot_topomap(am_nt_hbo[:,seed],raw_haemo_hbo.info,
            #         sphere=tansf_sp,extrapolate=extrapol,
            #         vlim=(vmin,vmax),
            #         cmap=mycm,
            #         names= ch_names_hbo_mne,
            #         mask=mask,mask_params=mask_par,
            #         size=5,axes=ax,contours=0)  
            # if thres>0:
            #     fig.suptitle(roi_label+'  '+subj+' top '+str((1-thres)*100)+'% FC',size='xx-large',weight='bold')
            # else:
            #     fig.suptitle(roi_label+'  '+subj,size='xx-large',weight='bold')
            # if j==0:
            #     ax.set_ylabel('HbO',fontsize=16)
            # if j== len(roi)-1:
            #     cb = plt.colorbar(im,label=cb_label,shrink=0.7)  
            #     if thres!=0:
            #         cb.set_ticks([vmin, -thres_hbo, 0, thres_hbo, vmax])
            #         cb.set_ticklabels(["{:.1f}".format(vmin), "{:.1f}".format(-thres_hbo), "0",  "{:.1f}".format(-thres_hbo), "{:.1f}".format(vmax)])
           
            if val=='z':
                vmin = -np.max(abs(am_nt_hbr))#[:,seed]))
                vmax = np.max(abs(am_nt_hbr))#[:,seed]))
                cb_label = "Fisher's z"
    
            else:
                vmin = -1
                vmax = 1
                cb_label = "Pearson's r"
    
            ax = fig.add_subplot(gs[sess,j])
            im,cn = mne.viz.plot_topomap(am_nt_hbr[:,seed],raw_haemo_hbr.info,
                    sphere=tansf_sp,extrapolate=extrapol,
                    vlim=(vmin,vmax),
                    cmap=mycm,
                    names= ch_names_hbr_mne,
                    mask=mask,mask_params=mask_par,
                    size=5,axes=ax,contours=0)
            if thres>0:
                fig.suptitle(roi_label+'  '+subj+' top '+str((1-thres)*100)+'% FC',size='xx-large',weight='bold')
            else:
                fig.suptitle(roi_label+'  '+subj,size='xx-large',weight='bold')            
            if j==0:
                ax.set_ylabel('HbR session '+str(sess+1),fontsize=16)
            if j== len(roi)-1:
                cb = plt.colorbar(im,label=cb_label,shrink=0.7)    
                if thres!=0:
                    cb.set_ticks([vmin, -thres_hbr, 0, thres_hbr, vmax])
                    cb.set_ticklabels(["{:.1f}".format(vmin), "{:.1f}".format(-thres_hbr), "0",  "{:.1f}".format(-thres_hbr), "{:.1f}".format(vmax)])
            mask[seed] = False
            j+=1
        if sqw >0:
            vmin = 0.1
            vmax = 0.95
            cb_label = "Quality"
            
            ax = fig.add_subplot(gs[sess,n_ch_roi])
            im,cn = mne.viz.plot_topomap(sq_nt_hbx[0,:],raw_haemo_hbr.info,
                    sphere=tansf_sp,extrapolate=extrapol,
                    vlim=(vmin,vmax),
                    cmap=mpl.colormaps['RdYlGn'],
                    names= ch_names_hbr_mne,
                    mask=mask,mask_params=mask_par,
                    size=5,axes=ax,contours=0)
            cb = plt.colorbar(im,label=cb_label,shrink=0.7)  

    for i in range(n_ch_roi):
        vmax = abs(A[:,i,:]).max()*1.1
        ax = fig.add_subplot(gs[2,i])
        #A[:,i,0] - A[:,i,1]
        if np.all(A[:,i,0]==0) or np.all(A[:,i,1]==0):
            if np.all(A[:,i,0]==0):
                plt.text(0.2,0.5,"Bad channel in session 1")
            if np.all(A[:,i,1]==0):                
                plt.text(0.2,0.6,"Bad channel in session 2")
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(A[:,i,0],A[:,i,1])
            #ypred = intercept + slope*A[:,i,0]
            #R_square = r2_score(A[:,i,1], ypred) 
            sns.regplot(x=A[:,i,0],y=A[:,i,1],color='blue', 
                        line_kws={'label':"R^2={:.2f}".format(r_value**2)},ax=ax)
            ax.legend()
            ax.set_xlim(-vmax,vmax)
            ax.set_ylim(-vmax,vmax)
            ax.set_ylabel('Session 2')
            ax.set_xlabel('Session 1')
            #ax.set_title(ch_names_hbr_mne[])
    return fig

    #fname_fig = bids_root.joinpath('results',all_scannames[isubj][0][0:-6]+'_SeedTopo_'+str(dur)+'.png')
    #fig.savefig(fname_fig, facecolor=fig.get_facecolor())
         

#%%
#from https://www.neurosynth.org/studies/26975555/
#mPFC_L = [[1,2],[2,2],[1,31],[2,31]]
#mPFC_R = [[15,16],[16,16],[15,44],[16,44]]
mPFC_L = [[1,31],[2,31],[29,30],[29,31],[29,32]]
mPFC_R = [[15,44],[16,44],[35,30],[35,44],[35,45]]
lIPL = [[10,11],[11,11],[11,12]]
rIPL = [[24,25],[25,25],[25,26]]
PCC_R = [[27,43],[28,43],[27,55],[41,55],[41,43]] 
PCC_L = [[14,43],[13,43],[13,42],[41,42]]
roi = mPFC_L
save_flag = 1
roi_labels = ["mPFCleft","mPFCright","IPLleft","IPLright","PCCleft","PCCright"]
roi_label = roi_labels[0]
bids_root = pathlib.Path(r'C:\Users\smontero\OneDrive - Boston University\RS_MovieWatching\Rest_Movie_WorkingMemory')
my_subjs=get_entity_vals(bids_root,'subject')
#sub = my_subjs[-1] #9
#sub = my_subjs[2] #11
#sub = my_subjs[3] #12
sess = "2"
thres = 0.0
val = 'z'
sqw = 1
for sub in my_subjs:
    bids_path = BIDSPath(subject=sub, task="RS", datatype="nirs",
                         root=bids_root, suffix="nirs",run=sess,
                         extension=".snirf")
    fileName = str(bids_path.fpath)
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    
    
    ## Fixing name and type convention
    channel_renaming_dict = {name: name.replace('-','_',1) for name in raw_intensity.ch_names}
    raw_intensity.rename_channels(channel_renaming_dict) 
    channel_renaming_dict = {name: name.replace('-',' ',1) for name in raw_intensity.ch_names}
    raw_intensity.rename_channels(channel_renaming_dict)  
    channel_retype_dict = {name: 'fnirs_cw_amplitude' for name in raw_intensity.ch_names}
    raw_intensity.set_channel_types(channel_retype_dict)
    #Fixing 2D coords
    get_2D_coords(raw_intensity)
    raw_intensity = get_long_channels(raw_intensity, min_dist=0.01)
    #OD
    raw_od = optical_density(raw_intensity)    
    #Hb
    raw_haemo = beer_lambert_law(raw_od, ppf=0.1)

    seedcorr_map_surf(raw_haemo,sub,mPFC_L,roi_labels[0],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[0]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())
    #plt.close('all')
    
    seedcorr_map_surf(raw_haemo,sub,mPFC_R,roi_labels[1],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[1]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())
    #plt.close('all')
    
    seedcorr_map_surf(raw_haemo,sub,lIPL,roi_labels[2],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[2]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())    
    #plt.close('all')
    
    
    seedcorr_map_surf(raw_haemo,sub,rIPL,roi_labels[3],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[3]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())
    #plt.close('all')
    
    seedcorr_map_surf(raw_haemo,sub,PCC_L,roi_labels[4],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[4]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())
    #plt.close('all')
    
    seedcorr_map_surf(raw_haemo,sub,PCC_R,roi_labels[5],val,thres,sqw)
    fig = plt.gcf()
    fname_fig = bids_root.joinpath('results','ch_surf_sub-'+sub+'_RS_runsDiff_'+roi_labels[5]+'_sqw-'+str(sqw)+'_val-'+val+'_thres-'+str(thres)+'.png')
    if save_flag == 1:
        fig.savefig(fname_fig, facecolor=fig.get_facecolor())
        plt.close('all')
    

#%%
# Download anatomical locations
subjects_dir = str(mne.datasets.sample.data_path()) + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw_intensity.info, trans='fsaverage', fnirs=['channels', 'sources', 'detectors'])
brain.show_view(azimuth=180, elevation=80, distance=450)

#%%
raw_ = raw_haemo.copy()
channel_renaming_dict = {name: str(i) for i, name in enumerate(raw_.ch_names)}
raw_.rename_channels(channel_renaming_dict) 
aa=mne.viz.plot_sensors(raw_.info,
                    pointsize=260,linewidth=0,show_names=True,
                    kind='select',
                    to_sphere=True)                  

print(aa)
del raw_