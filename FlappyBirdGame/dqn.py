import argparse
import sys
import random
import json
import numpy as np
import skimage as skimage
import tensorflow as tf

from skimage.transform import rotate
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
from collections import deque

from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

sys.path.append("game/")
import wrapped_flappy_bird as game

# SIMULATION PARAMS
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

class Agent:
	def __init__(self):
		pass

	def build_model(self):
		model = Sequential()
    	model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    	model.add(Activation('relu'))
	    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
	    model.add(Activation('relu'))
	    model.add(Flatten())
	    model.add(Dense(512))
	    model.add(Activation('relu'))
	    model.add(Dense(2))