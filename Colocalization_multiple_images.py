#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: keriabermudez

Version April 26, 2017

Script to calculate colocalization measures for several images and saves the marked images.
1	Separate into two channels (ch1 and ch2). Ch1 is the channel you would like to threshold and Ch2 is the one you want to measure
2	It will generate a channel mask based on a threshold using the Otsu Yen, or Isodata algorithms

    These are the measures:
        
        Ch1 and Ch2  Measurements:
            
        Th_Mth - Threshold method for that channel
        Threshold - Threshold value for that channel
        Sum_above_th - Sum of all the ch2 pixel intensities above threshold for that channel
        Area_above_th  - Number of channel pixels above threshold 
        Mean_above_th  - Mean intensity of channel pixels above threshold
        cmd -  Cumulative distribution for ch1 or ch2 above threshold
    
        Overlap Measurements:
            
        Overlap_Sum -  Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
        Overlap_Mean -  Mean of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
        M1 -  Mander's coefficient 1
        M2 -  Mander's coefficient 2
       
        Overlap_Area -  Area of overlaps
        Pearsons -   Pearson's coefficient for ch1 and ch2 intensities
        R-squared -  R**2 for ch1 and ch2 intensities


"""

import skimage.external.tifffile as tf
import numpy as np
import os
from pandas import DataFrame as df
import sys
from skimage import io
from skimage import segmentation
import mahotas as mh
import Colocalization_2D_Z_stacks as coloc_2d_z


local = True

if local: 
    # Path where you have the 2D or zstacks 
    path = '/Users/keriabermudez/Dropbox/Projects/Julia/Julia_Keria_Examples_82616/'
    # Path where you want the results to be saved
    path_results = path+'Results/'
    channel_1 = 'red'
    channel_2 = 'green'
    name = 'Results_5416'
    format_image = 'lsm'
    channel_1_th = 'isodata'
    channel_2_th = 'isodata'
    #cmd_limit = 0.5 #  if the threshld method results in a forground that covers more than 50% of the image, then use threshold values cmd_th_val as limit
    #cmd_th_val = 0.005
else:
    path = str(sys.argv[2])
    path_results = path+'Results/'
    channel_1 = str(sys.argv[3])
    channel_2 = str(sys.argv[4])
    name = str(sys.argv[5])
    format_image = str(sys.argv[6])
    channel_1_th = int(sys.argv[7])
    channel_2_th = int(sys.argv[8])
    
if not os.path.exists(path_results):
        os.makedirs(path_results)    

colors =  {'red':0,'green':1,'blue':2}
all_values = {}

for img in os.listdir(path):
    if img.endswith(format_image):
        image_file = path + img        
        image = coloc_2d_z.confocal_coloc(image_file,channel_1,channel_2, ch1_th=channel_1_th, ch2_th=channel_2_th)
        #Saving values
        img_vals = {}
        
        #channel 1
        img_vals[channel_1+'_Sum'] = image.ch1_sum # eliminate
        img_vals[channel_1+'_Th_Mth'] = image.ch1_mth
        img_vals[channel_1+'_Threshold'] = image.ch1_th
        img_vals[channel_1+'_Sum_above_th']= image.ch1_above_th
        img_vals[channel_1+'_Area_above_th']= image.ch1_above_th_count
        img_vals[channel_1+'_Mean_above_th']= image.ch1_above_th/image.ch1_above_th_count
        
        #channel 2
        img_vals[channel_2+'_Th_Mth'] = image.ch2_mth
        img_vals[channel_2+'_Threshold'] = image.ch2_th
        img_vals[channel_2+'_Sum_above_th']= image.ch2_above_th
        img_vals[channel_2+'_Area_above_th']= image.ch2_above_th_count
        img_vals[channel_2+'_Mean_above_th']= image.ch2_above_th/image.ch2_above_th_count
        
        img_vals[channel_2+'_Overlap_Sum'] = image.ch2_overlap_sum
        img_vals[channel_2+'_Overlap_Mean'] = image.ch2_overlap_mean 
        img_vals[channel_1+'_M1'] = image.M1
        img_vals[channel_2+'_M2'] = image.M2
        img_vals[channel_1+'_cmd'] = image.ch1_cmd
        img_vals[channel_2+'_cmd'] = image.ch2_cmd
        img_vals['Overlap_Area'] = image.overlap_area
        img_vals['Pearsons'] = image.pearsons()
        img_vals['R-squared'] = image.lineareg()
        all_values[img]= img_vals
        #Saving  Mask
        #
        ch1_mask_img = np.zeros(image.ch1_mask.shape, 'uint16') #check dtype
        ch1_mask_img[image.ch1_mask]= 65535
        ch2_mask_img = np.zeros(image.ch2_mask.shape, 'uint16') #check dtype
        ch2_mask_img[image.ch2_mask]= 65535
        overlap_mask_img = np.zeros(image.overlap_mask.shape, 'uint16') #check dtype
        overlap_mask_img[image.overlap_mask]= 65535
        
        tf.imsave(path_results+img[:-4]+'_'+channel_1+'_mask.tif', ch1_mask_img)
        tf.imsave(path_results+img[:-4]+'_'+channel_2+'_mask.tif', ch2_mask_img)
        tf.imsave(path_results+img[:-4]+'_'+'overlap'+'_mask.tif', overlap_mask_img)
        
        if image.len_shape == 3:
            color_1 = colors[channel_1]
            color_2 = colors[channel_2]
            
            color_image_ch1 = np.zeros((image.ch1_mask.shape[0],image.ch1_mask.shape[1],3), dtype= np.uint16)
            color_image_ch1[:,:,color_1]= image.ch1
            labeled, num_clusters= mh.label(ch1_mask_img, np.ones((3,3), bool))
            contours = np.zeros((image.ch1_mask.shape[0],image.ch1_mask.shape[1]), 'uint16')
            marked = segmentation.mark_boundaries(contours, labeled, color=[1,1,1], mode='outside')
            contours[marked[:,:,0] == 1] = image.ch1.max()
            color_image_ch1[:,:,2] = contours
            io.imsave(path_results+img[:-4]+'_'+channel_1+'.tif', color_image_ch1)

            color_image_ch2 = np.zeros((image.ch2_mask.shape[0],image.ch2_mask.shape[1],3), dtype= 'uint16')
            color_image_ch2[:,:,color_2]= image.ch2
            labeled, num_clusters= mh.label(ch2_mask_img, np.ones((3,3), bool))
            contours = np.zeros((image.ch2_mask.shape[0],image.ch2_mask.shape[1]), 'uint16')
            marked = segmentation.mark_boundaries(contours, labeled, color=[1,1,1], mode='outside')
            contours[marked[:,:,0] == 1] = image.ch2.max()
            color_image_ch2[:,:,2] = contours
            io.imsave(path_results+img[:-4]+'_'+channel_2+'.tif', color_image_ch2)

# Summary Table        
table = df(all_values)
table = table.T
table.to_csv(path_results+name+'.csv')



     

