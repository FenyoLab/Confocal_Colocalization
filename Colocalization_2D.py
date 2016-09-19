# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:26:58 2016

@author: keriabermudez
"""
"""
The script will read images or zstacks and will calculated several things.

For each image it will:

1	Separate into two channels (channel 1 and channel 2)
2	It will generate a channel mask based on a threshold using the Otsu algorithm.
3	After generating the mask it calculates several things:
◦	Channel 1 sum - which is the sum of all the channel 1 pixels intensities in the image
◦	Channel 1 threshold - which is the Otsu threshold value 
◦	Channel 1 sum above threshold - is the sum of all the chanel 1 pixel intensities above threshold
◦	Channel 1 count above Threshold - is the number of pixels above threshold
◦	Channel 2 sum - sum of all the channel 2 pixel intensities
◦	Channel 2 overlap sum - is the sum of the channel 2 pixel intensity values that overlap with channel 1 (these are the above threshold pixels)
◦	Channel 2 n overlap count - is the number of pixels that  don’t overlap with the channel 1
◦	Channel 2 n overlap sum - is the sum of the channel 2 pixel intensity values that don’t overlap with channel 1 (these are the above threshold pixels)
◦	Channel 2 overlap count - is the number of pixels that overlap with the channel 1
◦	Pearsons- pearsons coefficient for channel 1 and channel 2 intensities
◦     E-squared - R2 for channel 1 and channel 2 intensities
4	All the measures are outputted in cvs file and saved in path_results

All the channel 1 masks are saved in path_results

"""
from skimage.filters import threshold_otsu
import skimage.external.tifffile as tf
import numpy as np
import os
from pandas import DataFrame as df
from scipy import stats


class confocal_coloc:
    
    def __init__(self, image_dir, color_1, color_2):
        
      self.image_dir = image_dir
      self.color_1 = color_1
      self.color_2 = color_2
      full_image = tf.TiffFile(self.image_dir)
      image = full_image.asarray()
      if (len(image.shape) == 3):
          if(color_1 == 'red'):
            ch1 =  image[:,:,0] 
          if(color_1 == 'green'):
            ch1= image[:,:,1]
          if(color_1 == 'blue'):
            ch1 = image[:,:,2]   
      if (len(image.shape) == 3):
          if(color_2 == 'red'):
            ch2 =  image[:,:,0] 
          if(color_2 == 'green'):
            ch2= image[:,:,1]
          if(color_2 == 'blue'):
            ch2 = image[:,:,2]    
      if (len(image.shape) == 4):
          if(color_1 == 'red'):
            ch1 =  image[:,0,:,:] 
          if(color_1 == 'green'):
            ch1= image[:,1,:,:]
          if(color_1 == 'blue'):
            ch1 = image[:,2,:,:] 
      if (len(image.shape) == 4):
          if(color_2 == 'red'):
            ch2 =  image[:,0,:,:] 
          if(color_2 == 'green'):
            ch2= image[:,1,:,:]
          if(color_2 == 'blue'):
            ch2 = image[:,2,:,:] 
      
      self.ch1_th = threshold_otsu(ch1) 
      self.ch1_sum = ch1.sum()#Sum of all the ch1 pixels intensities  in the image
      self.ch1_mask = ch1 > self.ch1_th #Generate a ch1 mask based on a threshold using the Otsu algorithm.       
      self.ch1_above_th = ch1[self.ch1_mask].sum() #Sum of all the ch1 pixel intensities above threshold
      self.ch1_above_th_count = len(ch1[self.ch1_mask]) #Number of pixels above threshold
      self.ch2_sum = ch2.sum()#Sum of all the ch2 pixel intensities
      self.ch2_overlap_sum =  ch2[self.ch1_mask].sum()#Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_overlap_count = len(ch2[self.ch1_mask]) #Number of pixels that overlap with the ch1    
      self.ch2_n_overlap_sum =  ch2[~self.ch1_mask].sum() #Sum of the ch2 pixel intensity values that don’t overlap with the ch1  
      self.ch2_n_overlap_count = len(ch2[~self.ch1_mask]) #Number of pixels that  don’t overlap with the ch1
      self.ch1_flat = ch1.flatten()
      self.ch2_flat = ch2.flatten()
    
    def pearsons(self):
      pearsons = stats.pearsonr(self.ch1_flat,self.ch2_flat)
      return pearsons[0]
     
      
    def lineareg(self):
      lineareg = stats.linregress(self.ch1_flat,self.ch2_flat )
      return lineareg[2]**2

#%%

# Path where you have the 2D or zstacks 
path = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/For_Julia/Data/'
# Path where you wnat the results to be saved
path_results = '/Users/keriabermudez/Dropbox/BioIMGBlog/Projects/JuliaDerk/For_Julia/Masks/'

channel_1 = 'red'
channel_2 = 'green'

name = 'AD_run_3'
format_image = '.tif'

all_values = {}
for img in os.listdir(path):
    if img.endswith(format_image):
        image_file = path + img        
        image = confocal_coloc(image_file,channel_1,channel_2 )
        #Saving values
        img_vals = {}
        img_vals[channel_1+'_Sum'] = image.ch1_sum
        img_vals[channel_1+'_threshold'] = image.ch1_th
        img_vals[channel_1+'_Sum_above_th']= image.ch1_above_th
        img_vals[channel_1+'_Count_above_th']= image.ch1_above_th_count
        img_vals[channel_2+'_Overlap_Sum'] = image.ch2_overlap_sum
        img_vals[channel_2+'_Overlap_Count'] = image.ch2_overlap_count
        img_vals[channel_2+'_Outside_Overlap_Sum'] = image.ch2_n_overlap_sum
        img_vals[channel_2+'_Outside_Overlap_Sum'] = image.ch2_n_overlap_count
        img_vals['Pearsons'] = image.pearsons()
        img_vals['R-squared'] = image.lineareg()
        all_values[img]= img_vals
        #Saving  Mask
        mask_img = np.zeros(image.ch1_mask.shape, 'uint16')
        mask_img[image.ch1_mask]= 65535
        tf.imsave(path_results+img[:-4]+channel_1+'_mask_Otsu.tif', mask_img)
        

# Summary Table        
table = df(all_values)
table = table.T
table.to_csv(path_results+name+'.csv')

