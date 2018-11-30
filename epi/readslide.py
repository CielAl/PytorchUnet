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
	 
	


def defaultLabelMap(io):
	#0 1 2 3 --> -1 0 1 2
	io = io - 1;
	return io
#nameparser - return dict of func
def defaultNameParser(fname,params):
	imgdir = params.get('imgdir','.');
	suffix = params.get('suffix',"_mask.png")
	file_ext = params.get('file_ext','.png')
	fullname = imgdir+'/'+os.path.basename(fname).replace(suffix,file_ext)
	name = {}
	name['img'] = fullname
	name['mask'] = fname
	return name
	
def ioAdaptor(name,imgtype,inputfun,params,totals,**kwargs):
#inputfun-->function to read. Further abstraction or detail can be done outside of this wrapper 
#name-->parsed name from outside
	io = inputfun(name)
	if(imgtype=="mask"): #if its a mask image, then we only need a single channel (since grayscale 3D images are equal in all channels)
		if kwargs.get("labelChange",None) is not None:
			#function instead of table, in case there are more logical operations.
			io = kwargs["labelChange"](io)
		#label count
		for i,key in enumerate(params['classes']): #sum the number of pixels, this is done pre-resize, the but proportions don't change which is really what we're after
			totals[1,i]+=sum(sum(io[:,:,0]==key))			
	return io	