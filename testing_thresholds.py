#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:08:46 2017

@author: keriabermudez

Version April 26, 2017

Script to calculate the true negative rate and true positive rate of three threshold methods 'isodata','yen', and 'otsu' based on manual thresholds


"""

import skimage.external.tifffile as tf
import numpy as np
import os
from pandas import DataFrame as df
import  matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import Colocalization_2D_Z_stacks as coloc_2d_z

mpl.rcParams['xtick.labelsize'] = 7 
mpl.rcParams['ytick.labelsize'] = 7 
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42

sns.set()
sns.set_context("paper")
sns.set_style("ticks",{"xtick.major.size": 4, "ytick.major.size": 4,"ytick.major.direction": 'in',"xtick.major.direction": 'in'})

      
#%%%

path = '/Users/keriabermudez/Dropbox/Projects/Julia/Keria_ActaNeuropathologica_ManualThreshold/'
orig_table = pd.read_excel(path+'32217_ManualQuantification_Julia.xlsx',index_col =0)

#%%

# Path where you want the results to be saved
path_results = path+'Results_Julia/'
channel_1 = 'red'
channel_2 = 'green'
name = 'Results_5416'
format_image = 'lsm'

all_values = {}

th_list = ['isodata','otsu','yen']

n= 0
for img in orig_table.index:
    
    image_file = path + img    
    ch1_th_manual = orig_table.ix[img,'Red-Manual']
    ch2_th_manual = orig_table.ix[img,'Green-Manual']
    if not os.path.exists(image_file):
        print image_file + 'does not exist'
        continue
    
    #
    image_manual = coloc_2d_z.confocal_coloc(image_file,channel_1,channel_2, ch1_th=ch1_th_manual-1, ch2_th=ch2_th_manual-1)
   
    
    for method in th_list:
        img_vals = {}
        #channel 1
        img_vals['image']=img
        img_vals['color'] ='Red'
        img_vals['Manual']= ch1_th_manual
        img_vals['Area']= image_manual.ch1_above_th_count
        img_vals['Marker'] = orig_table.ix[img,'Red-Marker']
        img_vals['method'] =method
        
        image_method = coloc_2d_z.confocal_coloc(image_file,channel_1,channel_2, ch1_th=method, ch2_th=method)
        
        true_positive_mask = np.logical_and(image_method.ch1_mask,image_manual.ch1_mask)
        
        img_vals['true_positive_area'] = true_positive_mask.sum()
        img_vals['true_positive'] = true_positive_mask.sum()/float(image_manual.ch1_mask.sum())
        
        true_negative_mask = np.logical_and(~image_method.ch1_mask,~image_manual.ch1_mask)
        img_vals['true_negative_area'] = true_negative_mask.sum()
        
        manual_bg = ~image_manual.ch1_mask
        img_vals['true_negative'] = true_negative_mask.sum()/float(manual_bg.sum())
        img_vals['Negative_Area'] = float(manual_bg.sum())
        
        #Saving  Mask
        ch1_mask_img = np.zeros(image_manual.ch1_mask.shape, 'uint16') #check dtype
        ch1_mask_img[image_manual.ch1_mask]= 65535
        tf.imsave(path_results+'Red/manual/'+img[:-4]+'_manual_'+channel_1+'_mask.tif', ch1_mask_img)
        
        #Saving  Mask
        ch1_mask_img = np.zeros(image_manual.ch1_mask.shape, 'uint16') #check dtype
        ch1_mask_img[image_method.ch1_mask]= 65535
        tf.imsave(path_results+'Red/'+method+'/'+img[:-4]+'_'+method+'_'+channel_1+'_mask.tif', ch1_mask_img)
        
        all_values[n]= img_vals
        n+=1
    
    for method in th_list:
        img_vals = {}
        img_vals['image']=img
        img_vals['color'] ='Green'
        img_vals['Manual']= ch2_th_manual
        img_vals['Area']= image_manual.ch2_above_th_count
        img_vals['Marker'] = orig_table.ix[img,'Green-Marker']
        img_vals['method'] =method

        image_method = coloc_2d_z.confocal_coloc(image_file,channel_1,channel_2, ch1_th=method, ch2_th=method)
        true_positive_mask = np.logical_and(image_method.ch2_mask,image_manual.ch2_mask)
        
        img_vals['true_positive_area'] = true_positive_mask.sum()
        img_vals['true_positive'] = true_positive_mask.sum()/float(image_manual.ch2_mask.sum())
        
        true_negative_mask = np.logical_and(~image_method.ch2_mask,~image_manual.ch2_mask)
        img_vals['true_negative_area'] = true_negative_mask.sum()
        
        manual_bg = ~image_manual.ch2_mask
        img_vals['true_negative'] = true_negative_mask.sum()/float(manual_bg.sum())
        img_vals['Negative_Area'] = float(manual_bg.sum())
        
        #Saving  Mask
        ch2_mask_img = np.zeros(image_manual.ch2_mask.shape, 'uint16') #check dtype
        ch2_mask_img[image_manual.ch2_mask]= 65535
        tf.imsave(path_results+'Green/manual/'+img[:-4]+'_manual'+channel_2+'_mask.tif', ch2_mask_img)
        
        #Saving  Mask
        ch2_mask_img = np.zeros(image_manual.ch2_mask.shape, 'uint16') #check dtype
        ch2_mask_img[image_method.ch2_mask]= 65535
        tf.imsave(path_results+'Green/'+method+'/'+img[:-4]+'_'+method+'_'+channel_2+'_mask.tif', ch2_mask_img)
        all_values[n]= img_vals
        n+=1

table = df(all_values)
table = table.T
table.to_csv(path_results+'Results_Rearranged.csv')

#%%
green_table = table[table['color']=='Green']

methods= ['isodata','yen','otsu']

#%% Diaph1 Green

comparing_methods = {}
array_list_negative =[]
array_list_positive =[]

for method in methods:
    
    method_table = green_table[green_table['method']== method]
    true_neg_mean = method_table.true_negative.mean()
    true_neg_sd= method_table.true_negative.std()
    
    true_pos_mean = method_table.true_positive.mean()
    true_pos_sd= method_table.true_positive.std()
    
    array_list_negative.append(true_neg_mean)
    array_list_negative.append(true_neg_sd)
    
    array_list_positive.append(true_pos_mean)
    array_list_positive.append(true_pos_sd)
    
arrays = [['isodata','isodata','yen','yen','otsu','otsu'],['mean','sd','mean','sd','mean','sd','mean','sd']]
tuples = list(zip(*arrays))

index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Stats'])
negative = pd.Series(array_list_negative, index=index)
positive = pd.Series(array_list_positive, index=index)

diaph1_table_methods = df([negative,positive],index = ['True Negative Rate(Specificity)','True Positive Rate(Sensitivity)'])
diaph1_table_methods.to_csv(path_results+'Diaph1_table.csv')

#%% Markers Red ['Claudin5','alphaSMA','CD68','GFAP','MAP2','MBP']

red_table = table[table['color']=='Red']

markers  = ['Claudin5','alphaSMA','CD68','GFAP','MAP2','MBP']

for marker in  markers:
    marker_table = red_table[red_table['Marker'] == marker]
    
    array_list_negative =[]
    array_list_positive =[]
    
    
    for method in methods:
        
        method_table = marker_table[marker_table['method']== method]
        method_table.to_csv(path_results+marker+method+'_table.csv')
        
        true_neg_mean = method_table.true_negative.mean()
        true_neg_sd= method_table.true_negative.std()
        
        true_pos_mean = method_table.true_positive.mean()
        true_pos_sd= method_table.true_positive.std()
        
        array_list_negative.append(true_neg_mean)
        array_list_negative.append(true_neg_sd) 
        
        array_list_positive.append(true_pos_mean)
        array_list_positive.append(true_pos_sd)
        
    arrays = [['isodata','isodata','yen','yen','otsu','otsu'],['mean','sd','mean','sd','mean','sd','mean','sd']]
    tuples = list(zip(*arrays))

    index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Stats'])
    negative = pd.Series(array_list_negative, index=index)
    positive = pd.Series(array_list_positive, index=index)
        
    table_methods = df([negative,positive],index = ['True Negative Rate(Specificity)','True Positive Rate(Sensitivity)'])
    table_methods.to_csv(path_results+marker+'_table.csv')

#%% Plotting

fig, ax = plt.subplots(nrows = 2 ,ncols= 4,sharey = True,figsize = (6,4))

sns.stripplot(x='method', y='true_negative', data=green_table,ax = ax[0,0],size = 4 )
ax[0,0].set_title('DIAPH1')

red_table = table[(table['color']=='Red') & (table['Marker']=='Claudin5')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[0,1],size = 4)
ax[0,1].set_title('Claudin5')

red_table = table[(table['color']=='Red') & (table['Marker']=='alphaSMA')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[0,2],size = 4 )
ax[0,2].set_title('alphaSMA')

red_table = table[(table['color']=='Red') & (table['Marker']=='CD68')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[0,3] ,size = 4)
ax[0,3].set_title('CD68')

red_table = table[(table['color']=='Red') & (table['Marker']=='GFAP')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[1,0],size = 4 )
ax[1,0].set_title('GFAP')

red_table = table[(table['color']=='Red') & (table['Marker']=='MAP2')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[1,1] ,size = 4)
ax[1,1].set_title('MAP2')

red_table = table[(table['color']=='Red') & (table['Marker']=='MBP')]

sns.stripplot(x='method', y='true_negative', data=red_table,ax = ax[1,2],size = 4)
ax[1,2].set_title('MBP')

fig.tight_layout()

fig.savefig(path_results+'true_negative_rate.pdf')

#%%

fig, ax = plt.subplots(nrows = 2 ,ncols= 4,sharey = True,figsize = (6,4))

sns.stripplot(x='method', y='true_positive', data=green_table,ax = ax[0,0],size = 4 )
ax[0,0].set_title('DIAPH1')
red_table = table[table['color']=='Red']


red_table = table[(table['color']=='Red') & (table['Marker']=='Claudin5')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[0,1],size = 4 )
ax[0,1].set_title('Claudin5')

red_table = table[(table['color']=='Red') & (table['Marker']=='alphaSMA')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[0,2] ,size = 4)
ax[0,2].set_title('alphaSMA')

red_table = table[(table['color']=='Red') & (table['Marker']=='CD68')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[0,3] ,size = 4)
ax[0,3].set_title('CD68')

red_table = table[(table['color']=='Red') & (table['Marker']=='GFAP')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[1,0] ,size = 4)
ax[1,0].set_title('GFAP')

red_table = table[(table['color']=='Red') & (table['Marker']=='MAP2')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[1,1] ,size = 4)
ax[1,1].set_title('MAP2')

red_table = table[(table['color']=='Red') & (table['Marker']=='MBP')]

sns.stripplot(x='method', y='true_positive', data=red_table,ax = ax[1,2],size = 4)
ax[1,2].set_title('MBP')

fig.tight_layout()
fig.savefig(path_results+'true_positive_rate.pdf')
