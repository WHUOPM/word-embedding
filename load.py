import theano 
import theano.tensor as T 

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 

import numpy as np 

import pygame
from pygame.locals import * 

import cPickle
import time

import os 

def build_vocab(files):
    vocab_to_index = {}
    index_to_vocab = {}

    idx = 0
    total_words = 0
    for f in files: 
        fi = open(f, 'r')
        lines = fi.readlines()
        
        for line in lines:
            split_line = line.split(' ')
            for word in split_line:
                total_words += 1 
                if word not in vocab_to_index: 
                    vocab_to_index[word] = idx 
                    index_to_vocab[idx] = word 

                    idx += 1

        fi.close()

    return vocab_to_index, index_to_vocab

def tokenize(file, vocab_to_index):
    tokenized = []
    
    fi = open(f, 'r')
    lines = fi.readlines() 

    for line in lines:
        split_line = line.split(' ')
        for word in split_line:
            tokenized.append(vocab_to_index[word])
    
    fi.close()

    return tokenized 


