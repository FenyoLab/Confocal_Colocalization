#!/usr/bin/python



import cv2
import numpy as np
from skimage import io,feature,segmentation,exposure
import os
from scipy import ndimage 
import mahotas as mh
from skimage.measure import label,regionprops
from skimage.filters import threshold_otsu
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
import sys
from skimage.util.dtype import dtype_range
 
 
def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)

def plot_img_and_hist(img, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    fig1 ,ax = plt.subplots(nrows =1, ncols=1)
    # Display histogram
    ax.hist(img.ravel(), bins=bins)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[img.dtype.type]
    ax.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax.plot(bins, img_cdf, 'r')

    return fig1

def remove_regions(labeled_image, area, size='small'):
    labeled_img_copy = labeled_image.copy()
    if size == 'small': 
        for region in regionprops(labeled_img_copy):
            if region.area < area: #less than mean area - std or greater than
                for coord in region.coords:
                    labeled_img_copy[coord[0],coord[1]] = 0
        return labeled_img_copy
    else:
        for region in regionprops(labeled_img_copy):
            if region.area >= area: #less than mean area - std or greater than
                for coord in region.coords:
                    labeled_img_copy[coord[0],coord[1]] = 0
        return labeled_img_copy
        
def kmeans_img(cl1, K):
#cl1 = cv2.equalizeHist(frame[:,:,1])
#try k means
    
    #Z = cl1.reshape((-1))
    Z =  cl1.flatten()
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    if cl1.dtype == 'uint8':
        center = np.uint8(center)
    elif cl1.dtype == 'uint16':
        center = np.uint16(center)
    res = center[label.flatten()]
    res2 = res.reshape((cl1.shape))
    center  = np.sort(center, axis = 0)
    center = center[::-1]
    return res2, center

def watershedsegment(thresh,smooth_distance = True,kernel = 3):
    distances = mh.distance(thresh)
    #distances = distances/float(distances.ptp()) * 255 #
    #distance = ndimage.distance_transform_edt(thresh)
    if smooth_distance:
        distance = ndimage.gaussian_filter(distances, kernel)
    else:
        distance = distances
    
    
    maxima = feature.peak_local_max(distance, indices=False, exclude_border=False) #min_distance=10,
    surface = distance.max() - distance
    spots, t = mh.label(maxima) 
    areas, lines = mh.cwatershed(surface, spots, return_lines=True)
    
    labeled_clusters, num_clusters= mh.label(thresh, np.ones((3,3), bool))
          
    #join the 2 segmentation types - this takes care of separate clusters that are
    #labeled as one in the watershed, which can happen for a 'small' cluster near a 'large' cluster
    joined_labels = segmentation.join_segmentations(areas, labeled_clusters)
    labeled_nucl = joined_labels * thresh
    #labeled_nucl= remove_regions(labeled_nucl,3800)
    for index, intensity in enumerate(np.unique(labeled_nucl)):
            labeled_nucl[labeled_nucl == intensity] = index   
    
    lines[spots > 0] = False

    return labeled_nucl,lines