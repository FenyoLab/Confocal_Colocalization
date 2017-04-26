# -*- coding: utf-8 -*-

"""
@author: keriabermudez

Version April 26, 2017

The script will read images or zstacks and will calculated several things.

For each image it will:

1	Separate into two channels (ch1 and ch2). Ch1 is the channel you would like to threshold and Ch2 is the one you want to measure
2	It will generate a channel mask based on a threshold using the Otsu Yen, or Isodata algorithms
3	After generating the mask it calculates several things:
    
    ch1_cmd -  Cumulative distribution for ch1 above threshold
    ch2_cmd -  Cumulative distribution for ch2 above threshold
    ch1_sum - Sum of all the ch1 pixels intensities  in the image
    overlap_area - Area of overlaps
    ch1_above_th -  Sum of all the ch1 pixel intensities above threshold
    ch1_above_th_count -  Number of ch1 pixels above threshold
    ch2_sum - Sum of all the ch2 pixel intensities
    ch2_above_th - Sum of all the ch2 pixel intensities above threshold
    ch2_above_th_count - Number of ch2 pixels above threshold  
    ch2_overlap_sum -  Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
    ch2_overlap_mean - Mean of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
    ch2_n_overlap_sum - Sum of the ch2 pixel intensity values that don’t overlap with the ch1  
    ch2_n_overlap_count - Number of ch2  pixels that  don’t overlap with the ch1
    ch1_flat - 1D Array of ch1
    ch2_flat - 1D Array of ch2
    ch2_overlap_mask_sum -  Sum of ch2 intensities that are overlapping with ch1
    ch1_overlap_mask_sum -  Sum of ch1 intensities that are overlapping with ch2
    M1 - Mander's coefficient 1
    M2 -  Mander's coefficient 2
	pearsons - function that calculates pearsons coefficient for channel 1 and channel 2 intensities
    lineareg - function that calculates R**2 for channel 1 and channel 2 intensities

4   You can access the following masks

    ch1_mask - Ch1 mask based on a threshold using the selected algorithm.       
    ch2_mask - Ch2 mask based on a threshold using the selected algorithm. 
    overlap_mask - Overlap Mask

"""

from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata
import skimage.external.tifffile as tf
import numpy as np
from pandas import DataFrame as df
from scipy import stats
from skimage import  exposure

def get_lsm_ch(image_dir,color):
    """
    Getting the channel based on the color
    """
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

def cmd_th(image,perc): ## check this
    """
    Getting the threshold based on the percentage of coverage
    """
    if image.dtype == 'uint16':
        max_bin = 65536
    elif image.dtype == 'uint8':
         max_bin = 256
    img_cdf, bins = exposure.cumulative_distribution(image, max_bin)
    mask = img_cdf > perc
    return bins[mask].min()

def get_cmd(image,th):
    """
    Getting the  Cumulative distribution value 
    """
    if image.dtype == 'uint16':
        max_bin = 65536
    elif image.dtype == 'uint8':
         max_bin = 256
    img_cdf, bins = exposure.cumulative_distribution(image, max_bin)
    if np.any(bins == th):
        mask = bins == th
        cmd_val = img_cdf[mask][0]
    else:
        cmd_val = np.nan
    return cmd_val
    
class confocal_coloc:
    """
      Returns several colocalization values and masks

    """
    def __init__(self, image_dir, color_1, color_2,ch1_th = None, ch2_th = None):
     
      """
     
      Parameters
      ----------
      image_dir : string 
          Directory of image
      color_1 : string
          Color of channel 1. Example: 'red'
      color_2 : string
          Color of channel 2. Example: 'green'
      ch1_th : string or int
          Thresholding method or integer value to threshold ch1
      ch2_th : string or int
          Thresholding method or integer value to threshold ch2
    
      Returns
      -------
      self.ch1_th : int 
          Threshold value for ch1
      self.ch1_mth : String 
          Threshold method for ch1
      self.ch2_th : int 
          Threshold value for ch2
      self.ch2_mth : String 
          Threshold method for ch1
      self.ch1_cmd :  float
           Cumulative distribution for ch1 above threshold
      self.ch2_cmd :  float
           Cumulative distribution for ch2 above threshold
      self.ch1_sum : int
          Sum of all the ch1 pixels intensities  in the image
      self.ch1_mask :  ndarray
          Ch1 mask based on a threshold using the selected algorithm.       
      self.ch2_mask : ndarray
          Ch2 mask based on a threshold using the selected algorithm.       
      self.overlap_mask : ndarray
          Overlap Mask
      self.overlap_area : int
          Area of overlaps
      self.ch1_above_th : int
          Sum of all the ch1 pixel intensities above threshold
      self.ch1_above_th_count : int
          Number of ch1 pixels above threshold
      self.ch2_sum : int
          Sum of all the ch2 pixel intensities
      self.ch2_above_th : int
          Sum of all the ch2 pixel intensities above threshold
      self.ch2_above_th_count : int
          Number of ch2 pixels above threshold  
      self.ch2_overlap_sum : int
          Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_overlap_mean : float
          Mean of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_n_overlap_sum : int
          Sum of the ch2 pixel intensity values that don’t overlap with the ch1  
      self.ch2_n_overlap_count : int
          Number of ch2  pixels that  don’t overlap with the ch1
      self.ch1_flat : ndarray
          1D Array of ch1
      self.ch2_flat : ndarray
          1D Array of ch2
      self.ch2_overlap_mask_sum : int
          Sum of ch2 intensities that are overlapping with ch1
      self.ch1_overlap_mask_sum : int
          Sum of ch1 intensities that are overlapping with ch2
      self.M1 : float
          Mander's coefficient 1
      self.M2: float
          Mander's coefficient 2
    
      """
      
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
      self.len_shape = len_shape
      self.ch2 = ch2
      self.ch1 = ch1
      self.ch1_mth =  ch1_th
      self.ch2_mth =  ch2_th
      
      #channel 1
      if ch1_th == None or ch1_th == 'otsu':
          self.ch1_mth =  'otsu'
          self.ch1_th = threshold_otsu(ch1)
      elif ch1_th == 'yen':
          self.ch1_th = threshold_yen(ch1)
      elif ch1_th == 'isodata':
          self.ch1_th = threshold_isodata(ch1)
      else:
          self.ch1_th = ch1_th
      
      
      #channel 2
      if ch2_th == None or ch2_th == 'otsu':
          self.ch2_th = threshold_otsu(ch2)
          self.ch2_mth =  'otsu'
      elif ch2_th == 'yen':
          self.ch2_th = threshold_yen(ch2)
      elif ch2_th == 'isodata':
          self.ch2_th = threshold_isodata(ch2)
      else:
          self.ch2_th = ch2_th
      
      
      
      self.ch1_cmd = get_cmd(self.ch1, self.ch1_th) # Cumulative distribution for ch1
      self.ch2_cmd = get_cmd(self.ch2, self.ch2_th) # Cumulative distribution for ch2
      self.ch1_sum = ch1.sum() #Sum of all the ch1 pixels intensities  in the image
      self.ch1_mask = ch1 > self.ch1_th #Generate a ch1 mask based on a threshold using the selected algorithm.       
      self.ch2_mask = ch2 > self.ch2_th #Generate a ch2 mask based on a threshold using the selected algorithm.       
      self.overlap_mask = np.logical_and(self.ch1_mask,self.ch2_mask) 
      self.overlap_area = self.overlap_mask.sum()
      self.ch1_above_th = ch1[self.ch1_mask].sum() #Sum of all the ch1 pixel intensities above threshold
      self.ch1_above_th_count = len(ch1[self.ch1_mask]) #Number of pixels above threshold
      self.ch2_sum = ch2.sum() #Sum of all the ch2 pixel intensities
      self.ch2_above_th = ch2[self.ch2_mask].sum() #Sum of all the ch2 pixel intensities above threshold
      self.ch2_above_th_count = len(ch2[self.ch2_mask]) #Number of pixels above threshold  
      self.ch2_overlap_sum =  ch2[self.ch1_mask].sum() #Sum of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_overlap_mean =  ch2[self.ch1_mask].sum()/self.ch1_above_th_count #Mean of the ch2 pixel intensity values that overlap with the ch1 (these are the above threshold pixels)
      self.ch2_n_overlap_sum =  ch2[~self.ch1_mask].sum() #Sum of the ch2 pixel intensity values that don’t overlap with the ch1  
      self.ch2_n_overlap_count = len(ch2[~self.ch1_mask]) #Number of ch2  pixels that  don’t overlap with the ch1
      self.ch1_flat = ch1.flatten() 
      self.ch2_flat = ch2.flatten()
      self.ch2_overlap_mask_sum = ch2[self.overlap_mask].sum()
      self.ch1_overlap_mask_sum = ch1[self.overlap_mask].sum()
      self.M1 = self.ch1_overlap_mask_sum/float(self.ch1_above_th) # Mander's coefficient 1
      self.M2 = self.ch2_overlap_mask_sum/float(self.ch2_above_th) # Mander's coefficient 2
      

    def pearsons(self):
      pearsons = stats.pearsonr(self.ch1_flat,self.ch2_flat)
      return pearsons[0]
     
      
    def lineareg(self):
      lineareg = stats.linregress(self.ch1_flat,self.ch2_flat )
      return lineareg[2]**2

