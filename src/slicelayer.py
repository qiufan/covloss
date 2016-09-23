# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
   The layer combines the input image into triplet.Priority select the semi-hard samples
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math
import config

class SliceLayer(caffe.Layer):
        
    def setup(self, bottom, top):
        """Setup the TripletSelectLayer."""
        self.pairs = config.batch_size/2
        top[0].reshape(self.pairs,shape(bottom[0].data)[1])
        top[1].reshape(self.pairs,shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        top_left = []
        top_right = []

        for i in range(self.pairs): 
            top_left.append(bottom[0].data[2*i])
            top_right.append(bottom[0].data[2*i+1])

        top[0].data[...] = np.array(top_left).astype(float32)
        top[1].data[...] = np.array(top_right).astype(float32)
    

    def backward(self, top, propagate_down, bottom):
        for i in range(self.pairs):
	    bottom[0].diff[2*i] = top[0].diff[i]
            bottom[0].diff[2*i+1] = top[1].diff[i]

        #print 'backward-no_re:',bottom[0].diff[0][0]
        #print 'tripletlist:',self.no_residual_list

        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass







