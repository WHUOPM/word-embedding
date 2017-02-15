import theano 
import theano.tensor as T 

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 

import numpy as np 

import pygame
from pygame.locals import * 

import cPickle
import time

import os 

import load

files_to_open = []
for book in os.listdir('OUP'):
    for f in os.listdir('OUP/' + book):
        if f.endswith('.txt'):
            files_to_open.append('OUP/' + book + '/' + f)

def onehot(index, size):
    o = np.zeros(shape = size)
    o[index] = 1

    return o

def init_weights(shape1, shape2):
    shape = [shape1, shape2]
    return theano.shared(np.random.randn(*shape) * 0.01)

X = []  # size wordsize x vocabsize, onehot vectors

vocab_size = 300000
embed_size = 200

embed_param = init_weights(vocab_size, embed_size)
w1_param = init_weights(embed_size, vocab_size)
w2_param = init_weights(embed_size, vocab_size)
w3_param = init_weights(embed_size, vocab_size)
w4_param = init_weights(embed_size, vocab_size)

inp = T.matrix()   # batchsize x vocab_size_onehot 
target = T.matrix() # batchsize x output_onehotappended

def embedding(inp):
    embed_vec = T.dot(inp, embed_param)

    w1 = T.nnet.softmax(T.dot(embed_vec, w1_param))
    w2 = T.nnet.softmax(T.dot(embed_vec, w2_param))
    w3 = T.nnet.softmax(T.dot(embed_vec, w3_param))
    w4 = T.nnet.softmax(T.dot(embed_vec, w4_param))

    output = T.concatenate([w1, w2, w3, w4], axis = 1)  # batchsize x vocab_size 

    return embed_vec, output

embedding, context = embedding(inp)
loss = T.mean(T.sum(T.log(context * target), axis = 1) + T.sum(T.log((1 - context) * (1 - target)), axis = 1))  # log likelihood loss 
 
# Should implement RMSprop for this 
gradients = T.grad(cost = loss, wrt = [embed_param, w1_param, w2_param, w3_param, w4_param])

updates = []
for p, g in zip([embed_param, w1_param, w2_param, w3_param, w4_param], gradients):
    updates.append([p, p - g * 0.1])

train = theano.function(inputs = [inp, target], outputs = [loss, embedding], updates = updates)

print "start"
v2i, i2v = load.build_vocab(files_to_open)
tokenize = load.tokenize(files_to_open[0], v2i)

epochs = 1
print "start"

for epoch in range (epochs):
    for files in files_to_open: 
        
        tokenize = load.tokenize(files, v2i)
        print tokenize[0:100]