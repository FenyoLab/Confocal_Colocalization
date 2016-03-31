# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:26:58 2016

@author: keriabermudez
"""

from skimage.filters import threshold_otsu
import skimage.external.tifffile as tf
import numpy as np
import os
from pandas import DataFrame as df
from scipy import stats

# Path where you have the z-stacks
path = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/Data/'

# Path where you want the masks and the results table to be saved
path_results = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/Masks/'
name = 'AD_run_1'

all_values = {}
for img in os.listdir(path):
    if img.endswith(".tif"):
        image_file = path + img
        full_image = tf.TiffFile(image_file)
        image = full_image.asarray()
        #Separate the red and green channel
        red = image[:,0,:,:]
        green = image[:,1,:,:]
        #Sum of all the red pixels intensities  in the z-stack
        red_sum = red.sum() 
        #Generate a red mask based on a threshold using the Otsu algorithm.
        r_th = threshold_otsu(red)
        red_mask = red > r_th
        #Sum of all the red pixel intensities above threshold
        red_above_th = red[red_mask].sum()
        #Number of pixels above threshold
        red_above_th_count = len(red[red_mask])
        #Sum of all the green pixel intensities
        green_sum = green.sum()
        #Sum of the green pixel intensity values that overlap with the red (these are the above threshold pixels)
        green_overlap_sum =  green[red_mask].sum()
        #Number of pixels that overlap with the red
        green_overlap_count = len(green[red_mask])
        #Sum of the green pixel intensity values that don’t overlap with the red
        green_n_overlap_sum =  green[~red_mask].sum()
        #Number of pixels that  don’t overlap with the red
        green_n_overlap_count = len(green[~red_mask])
        # Prearsons Calculation
        red_flat = red.flatten()
        green_flat = green.flatten()
        pearsons =stats.pearsonr(red_flat,green_flat)
        
        #Saving values
        img_vals = {}
        img_vals['Red_Sum'] = red_sum
        img_vals['Red_threshold'] = r_th
        img_vals['Red_Sum_above_th']= red_above_th
        img_vals['Red_Count_above_th']= red_above_th_count
        img_vals['Green_Overlap_Sum'] = green_overlap_sum
        img_vals['Green_Overlap_Count'] = green_overlap_count
        img_vals['Green_n_Overlap_Sum'] = green_n_overlap_sum
        img_vals['Green_n_Overlap_Count'] = green_n_overlap_count
        img_vals['Pearsons'] = pearsons[0]
        all_values[img]= img_vals
        #Savinfg Red Mask
        red_mask_img = np.zeros(red.shape, 'uint16')
        red_mask_img[red_mask]= 65535
        tf.imsave(path_results+img[:-4]+'red_mask_Otsu.tif', red_mask_img)
        

# Summary Table        
table = df(all_values)
table = table.T
table.to_csv(path_results+name+'.csv')


