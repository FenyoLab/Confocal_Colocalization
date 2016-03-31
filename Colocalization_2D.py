# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:26:58 2016

@author: keriabermudez
"""
"""
The script will read images and will calculated several things.
For each image it:

1	Will separate the red and blue channel
2	It will generate a red mask based on a threshold using the Otsu algorithm. I tried another method but it came out the same, so that is why I recommend to use Otsu
3	After generating the mask it calculates several things:
◦	Red Sum- which is the sum of all the red pixels intensities in the image
◦	Red threshold- which is the Otsu threshold value 
◦	Red sum above threshold- is the sum of all the red pixel intensities above threshold
◦	Red count above Threshold- is the number of pixels above threshold
◦	Blue sum- sum of all the blue pixel intensities
◦	Blue overlap sum - is the sum of the blue pixel intensity values that overlap with the red (these are the above threshold pixels)
◦	Blue n overlap count - is the number of pixels that  don’t overlap with the red
◦	Blue n overlap sum - is the sum of the blue pixel intensity values that don’t overlap with the red (these are the above threshold pixels)
◦	Blue overlap count - is the number of pixels that overlap with the red
◦	Pearsons- pearsons coefficient for red and blue intensities
4	All the measures are outputted in cvs file and saved in path_results
All the red masks are saved in path_results

"""
from skimage.filters import threshold_otsu
import skimage.external.tifffile as tf
import numpy as np
import os
from pandas import DataFrame as df
from scipy import stats
from skimage import io

# Path where you have the 2D images
path = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/Confocal_GFAP_2D/'

# Path where you want the masks and the results table to be saved
path_results = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/Confocal_GFAP_2D/Masks/'
name = 'AD_run_1'

all_values = {}
for img in os.listdir(path):
    if img.endswith(".tif"):
        image_file = path + img
        image = io.imread(image_file)
        #Separate the green and blue channel
        green = image[:,:,1]
        blue = image[:,:,2]
        #Sum of all the green pixels intensities  in the image
        green_sum = green.sum() 
        #Generate a green mask based on a threshold using the Otsu algorithm.
        g_th = threshold_otsu(green)
        green_mask = green > g_th
        #Sum of all the green pixel intensities above threshold
        green_above_th = green[green_mask].sum()
        #Number of pixels above threshold
        green_above_th_count = len(green[green_mask])
        #Sum of all the blue pixel intensities
        blue_sum = blue.sum()
        #Sum of the blue pixel intensity values that overlap with the green (these are the above threshold pixels)
        blue_overlap_sum =  blue[green_mask].sum()
        #Number of pixels that overlap with the green
        blue_overlap_count = len(blue[green_mask])
        #Sum of the blue pixel intensity values that don’t overlap with the green
        blue_n_overlap_sum =  blue[~green_mask].sum()
        #Number of pixels that  don’t overlap with the green
        blue_n_overlap_count = len(blue[~green_mask])
        # Prearsons Calculation
        green_flat = green.flatten()
        blue_flat = blue.flatten()
        pearsons =stats.pearsonr(green_flat,blue_flat)
        
        #Saving values
        img_vals = {}
        img_vals['green_Sum'] = green_sum
        img_vals['green_threshold'] = g_th
        img_vals['green_Sum_above_th']= green_above_th
        img_vals['green_Count_above_th']= green_above_th_count
        img_vals['blue_Overlap_Sum'] = blue_overlap_sum
        img_vals['blue_Overlap_Count'] = blue_overlap_count
        img_vals['blue_n_Overlap_Sum'] = blue_n_overlap_sum
        img_vals['blue_n_Overlap_Count'] = blue_n_overlap_count
        img_vals['Pearsons'] = pearsons[0]
        all_values[img]= img_vals
        #Savinfg green Mask
        green_mask_img = np.zeros(green.shape, 'uint16')
        green_mask_img[green_mask]= 65535
        tf.imsave(path_results+img[:-4]+'green_mask_Otsu.tif', green_mask_img)
        

# Summary Table        
table = df(all_values)
table = table.T
table.to_csv(path_results+name+'.csv')


