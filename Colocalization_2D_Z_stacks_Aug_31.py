# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:26:58 2016

@author: keriabermudez
"""
"""
The script will read images or zstacks and will calculated several things.

For each image it will:

1	Separate into two channels (channel 1 and channel 2). Channel 1 is the channel you would like to threshold and channel 2 is the one you want to measure
2	It will generate a channel mask based on a threshold using the Otsu algorithm.
3	After generating the mask it calculates several things:
◦	Channel 1 sum - which is the sum of all the channel 1 pixels intensities in the image
◦	Channel 1 threshold - which is the Otsu threshold value for channel 1
◦	Channel 2 threshold - which is the Otsu threshold value for channel 2
◦	Channel 1 sum above threshold - is the sum of all the chanel 1 pixel intensities above threshold
◦	Channel 1 mean above threshold - is the mean of all the chanel 1 pixel intensities above threshold
◦	Channel 1 area above Threshold - is the number of pixels above threshold
◦	Channel 2 overlap sum - is the sum of the channel 2 pixel intensity values that overlap with channel 1 (these are the above threshold pixels)
◦	Channel 2 overlap mean - is the mean channel 2 pixel intensity values that overlap with channel 1 (these are the above threshold pixels)
◦	Channel 2 M - is the Manders M2 wich is calculate by the sum of the channel 2 pixel intensity values that  overlap with channel 1 (these are the above threshold pixels) divided by the sum of all channel 2 pixel intensities
◦	Overlap area - is the number of pixels that result from an and operation between the channel 1 mask and channel 2 mask
◦	Pearsons- pearsons coefficient for channel 1 and channel 2 intensities
◦     R-squared - R2 for channel 1 and channel 2 intensities
4	All the measures are outputted in cvs file and saved in path_results

All the channel 1 masks are saved in path_results

"""
from skimage.filters import threshold_otsu
import skimage.external.tifffile as tf
from skimage.measure import regionprops
import numpy as np
import os
from pandas import DataFrame as df
from scipy import stats
import sys

def get_lsm_ch(image_dir,color):

    full_image = tf.TiffFile(image_dir)
    is_lsm = full_image.is_lsm
    assert(is_lsm),"File is not lsm"
    all_pages =  full_image.pages
    page_1 = all_pages[0]
         
    lsm_info =  page_1.cz_lsm_scan_info
    channels = lsm_info.tracks
    channels_dict = {}
    for n in range(0, len(channels)):
        
        channels_dict[n] = channels[n].data_channels[0]
          
    table= df(channels_dict)
    
    colors = {}
    colors['blue'] = 16711680
    colors['red']= 255
    colors['green'] = 65280
    colors['dark_green'] = 48896
    channel = table.ix['acquire',:][table.ix['color',:] == colors[color]]
    
    if len(channel) == 0:
        channel = table.ix['acquire',:][table.ix['color',:] == colors['dark_green']]
        if len(channel) == 0:
            print "Error: No Color\n" 
            return None
        else:
            return channel.index[0]
    else:
        return channel.index[0]

class confocal_coloc:
    
    def __init__(self, image_dir, color_1, color_2,ch1_th = None, ch2_th = None):
      
      self.image_dir = image_dir
      self.color_1 = color_1
      self.color_2 = color_2
      full_image = tf.TiffFile(self.image_dir)
      
      image = full_image.asarray()
      format_img = self.image_dir[-3:]              
      is_lsm = full_image.is_lsm
      is_image_j = full_image.is_imagej
      is_tif = format_img == 'tif'
      len_shape = len(image.shape)
      tif_colors =  {'red':0,'green':1,'blue':2}
      
      if is_tif:   
          color_1_channel =  tif_colors[self.color_1]
          color_2_channel =  tif_colors[self.color_2]
      
      #Tiff and not Image J 
          if is_image_j: #Tiff and not Image J 2D
               if len_shape == 3: #Image J 2D
                    ch1 =  image[color_1_channel,:,:] 
                    ch2 =  image[color_2_channel,:,:] 
               elif len_shape == 4: #Image J zstack
                    ch1 =  image[:,color_1_channel,:,:] 
                    ch2 =  image[:,color_2_channel,:,:] 
          else: # not Jmage J 
              if len_shape == 3: # Not Image J 2D
                    ch1 =  image[:,:,color_1_channel] 
                    ch2 =  image[:,:,color_2_channel] 
              elif len_shape == 4: #Not Image J zstack
                    ch1 =  image[:,color_1_channel,:,:] 
                    ch2 =  image[:,color_2_channel,:,:]    
      elif is_lsm:
          color_1_channel =  get_lsm_ch(self.image_dir,self.color_1)
          color_2_channel =  get_lsm_ch(self.image_dir,self.color_2)
          if len_shape == 3:
              ch1 =  image[color_1_channel,:,:] 
              ch2 =  image[color_2_channel,:,:] 
          elif len_shape == 5:
              ch1 = image[0,:,color_1_channel,:,:]
              ch2 = image[0,:,color_2_channel,:,:]
           
      self.ch2 = ch2
      self.ch1 = ch1
      
      if ch1_th == None:
          self.ch1_th = threshold_otsu(ch1)
      else:
          self.ch1_th = ch1_th
      if ch2_th == None:
          self.ch2_th = threshold_otsu(ch2)
      else:
          self.ch2_th = ch2_th
          

      self.ch1_sum = ch1.sum() #Sum of all the ch1 pixels intensities  in the image
      self.ch1_mask = ch1 > self.ch1_th #Generate a ch1 mask based on a threshold using the Otsu algorithm.       
      self.ch2_mask = ch2 > self.ch2_th #Generate a ch2 mask based on a threshold using the Otsu algorithm.       
      self.overlap_mask = np.logical_and(self.ch1_mask,self.ch2_mask)
      self.overlap_area = self.overlap_mask.sum()
      self.ch1_above_th = ch1[self.ch1_mask].sum() #Sum of all the ch1 pixel intensities above threshold
      self.ch1_above_th_count = len(ch1[self.ch1_mask]) #Number of pixels above threshold
      self.ch2_sum = ch2.sum() #Sum of all the ch2 pixel intensities
      self.ch2_overlap_sum =  ch2[self.ch1_mask].sum() #Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_overlap_mean =  ch2[self.ch1_mask].sum()/self.ch1_above_th_count #Mean of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_n_overlap_sum =  ch2[~self.ch1_mask].sum() #Sum of the ch2 pixel intensity values that don’t overlap with the ch1  
      self.ch2_n_overlap_count = len(ch2[~self.ch1_mask]) #Number of ch2  pixels that  don’t overlap with the ch1
      self.ch1_flat = ch1.flatten()
      self.ch2_flat = ch2.flatten()
    
    def pearsons(self):
      pearsons = stats.pearsonr(self.ch1_flat,self.ch2_flat)
      return pearsons[0]
     
      
    def lineareg(self):
      lineareg = stats.linregress(self.ch1_flat,self.ch2_flat )
      return lineareg[2]**2

#%%
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
    channel_1_th = None
    channel_2_th = 300

else:
    path = str(sys.argv[2])
    channel_1 = str(sys.argv[3])
    channel_2 = str(sys.argv[4])
    name = str(sys.argv[5])
    format_image = str(sys.argv[6])
    channel_1_th = int(sys.argv[7])
    channel_2_th = int(sys.argv[8])

if not os.path.exists(path_results):
        os.makedirs(path_results)    


all_values = {}

for img in os.listdir(path):
    if img.endswith(format_image):
        image_file = path + img        
        image = confocal_coloc(image_file,channel_1,channel_2, channel_1_th, channel_2_th)
        #Saving values
        img_vals = {}
        img_vals[channel_1+'_Sum'] = image.ch1_sum # eliminate
        img_vals[channel_1+'_Threshold'] = image.ch1_th
        img_vals[channel_1+'_Sum_above_th']= image.ch1_above_th
        img_vals[channel_1+'_Area_above_th']= image.ch1_above_th_count
        img_vals[channel_1+'_Mean_above_Th']= image.ch1_above_th/image.ch1_above_th_count
        img_vals[channel_2+'_Threshold'] = image.ch2_th
        img_vals[channel_2+'_Overlap_Sum'] = image.ch2_overlap_sum
        img_vals[channel_2+'_Overlap_Mean'] = image.ch2_overlap_mean 
        img_vals[channel_2+'_M'] = image.ch2_overlap_sum/image.ch2_sum
        img_vals['Overlap_Area'] = image.overlap_area
        img_vals['Pearsons'] = image.pearsons()
        img_vals['R-squared'] = image.lineareg()
        all_values[img]= img_vals
        #Saving  Mask
        ch1_mask_img = np.zeros(image.ch1_mask.shape, 'uint16') #check dtype
        ch1_mask_img[image.ch1_mask]= 65535
        ch2_mask_img = np.zeros(image.ch2_mask.shape, 'uint16') #check dtype
        ch2_mask_img[image.ch2_mask]= 65535
        overlap_mask_img = np.zeros(image.overlap_mask.shape, 'uint16') #check dtype
        overlap_mask_img[image.overlap_mask]= 65535
        
        tf.imsave(path_results+img[:-4]+'_'+channel_1+'_mask.tif', ch1_mask_img)
        tf.imsave(path_results+img[:-4]+'_'+channel_2+'_mask.tif', ch2_mask_img)
        tf.imsave(path_results+img[:-4]+'_'+'overlap'+'_mask.tif', overlap_mask_img)

# Summary Table        
table = df(all_values)
table = table.T
table.to_csv(path_results+name+'.csv')

