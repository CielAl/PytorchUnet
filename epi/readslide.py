#ectracted from instance method getImgThumb of BaseImage.py(HistoQC Module)
import os
import numpy as np
import inspect
from distutils.util import strtobool
import openslide
import cv2
import openslide
from openslide import OpenSlide
import matplotlib.pyplot as plt


def os_get_best_down(osh,down_factor):
	level = osh.get_best_level_for_downsample(down_factor)
	relative_down = down_factor / osh.level_downsamples[level]
	win_size = 2048
	win_size_down = int(win_size * 1 / relative_down)
	dim_base = osh.level_dimensions[0]
	output = []
	for x in range(0, dim_base[0], round(win_size * osh.level_downsamples[level])):
		row_piece = []
		for y in range(0, dim_base[1], round(win_size * osh.level_downsamples[level])):
			aa = osh.read_region((x, y), level, (win_size, win_size))
			bb = aa.resize((win_size_down, win_size_down))
			row_piece.append(bb)
		row_piece = np.concatenate(row_piece, axis=0)[:, :, 0:3]
		output.append(row_piece)
	output = np.concatenate(output, axis=1)
	output = output[0:round(dim_base[1] * 1 / down_factor), 0:round(dim_base[0] * 1 / down_factor), :]
	return output

def getThumbByMag(osh,target_mag=10,base_mag=20,speed=False):
	if(speed):
		down_factor =  target_mag/base_mag
		new_dim = np.asarray(osh.dimensions) *down_factor
		output = np.array(osh.get_thumbnail(new_dim))
	else:
		down_factor =  base_mag / target_mag
		output = os_get_best_down(osh,down_factor)
	return output
	 
	
#simplify the args by composite types
def ndpiAdaptor(imgtype,imgdir,fname,file_ext,classes,mags,totals):
	fullname = imgdir+'/'+os.path.basename(fname).replace("_mask.png",file_ext)
	if(imgtype=="img"): #if we're looking at an img, it must be 3 channel, but cv2 won't load it in the correct channel order, so we need to fix that
		#print("imgMode:"+fullname)
		osh = OpenSlide(fullname)
		io= getThumbByMag(osh,mags[0],mags[1])
	else: #if its a mask image, then we only need a single channel (since grayscale 3D images are equal in all channels)
		#print("maskMode:"+fname)
		#io=cv2.imread(fname)[:,:,0]# multiclass mask
		io = plt.imread(fname)# if use plt -> single channel img will not have the 3rd axis.[:,:,0]
		for i,key in enumerate(classes): #sum the number of pixels, this is done pre-resize, the but proportions don't change which is really what we're after
			totals[1,i]+=sum(sum(io==key))
	return io
	
#simplify the args by composite types
#return 3channel img for both mask and img ->simplify the case of shape and size for padding and patch extraction
def pngAdaptor(imgtype,imgdir,fname,file_ext,classes,mags,totals):
	fullname = imgdir+'/'+os.path.basename(fname).replace("_mask.png",file_ext)
	if(imgtype=="img"): #if we're looking at an img, it must be 3 channel, but cv2 won't load it in the correct channel order, so we need to fix that
		#print("imgMode:"+fullname)
		io = plt.imread(fullname)
	else: #if its a mask image, then we only need a single channel (since grayscale 3D images are equal in all channels)
		#print("maskMode:"+fname)
		#io = (plt.imread(fname)*255).astype(np.uint8)# if use plt -> single channel img will not have the 3rd axis.[:,:,0]
		io = cv2.imread(fname)
		for i,key in enumerate(classes): #sum the number of pixels, this is done pre-resize, the but proportions don't change which is really what we're after
		#those semantic coupling is freakingly crazy
			totals[1,i]+=sum(sum(io[:,:,0]==key))
	return io	