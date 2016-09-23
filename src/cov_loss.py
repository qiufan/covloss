# --------------------------------------------------------
# min-max LOSS
# Copyright (c) 2016 qiufan Tech.
# Shi W, Gong Y, Wang J. Improving CNN Performance with Min-Max Objective[J].
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""

import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing

class CovLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the MinMaxLossLayer."""
	#layer_params = yaml.load(self.param_str)
        #self.k1 = layer_params['k1']
	#self.k2 = layer_params['k2']

        top[0].reshape(1)

    def forward(self, bottom, top):
	size=bottom[0].num
	self.dis_vector=np.zeros((size))
	for i in range(size):
	    x1=bottom[0].data[i]##pay attention
	    x2=bottom[1].data[i]
	    x1_x2=x1-x2
	    dis=np.dot(x1_x2,x1_x2)
	    self.dis_vector[i]=dis
	self.mean=np.mean(self.dis_vector)
	Loss=np.cov(self.dis_vector)
	Loss=Loss/bottom[0].num
	top[0].data[...] = Loss
	
        
    def backward(self, top, propagate_down, bottom):
	size=bottom[0].num        
	if propagate_down[0]:
	    for i in range(size):
	        bottom[0].diff[i]=(4*(self.dis_vector[i]-self.mean)*(bottom[0].data[i]-bottom[1].data[i])/size)
		bottom[1].diff[i]=(-4*(self.dis_vector[i]-self.mean)*(bottom[0].data[i]-bottom[1].data[i])/size)
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
