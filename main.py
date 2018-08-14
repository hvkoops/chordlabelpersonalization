# -*- coding: utf-8 -*-
from src.common import *
from src.plotting import *
import mir_eval
import h5py
import tensorflow as tf
from sklearn import model_selection
import sys
import matplotlib.pyplot as plt
import string
from time import gmtime, strftime
from sklearn.metrics import r2_score
import os
from glob import glob
from tabulate import tabulate
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.layers import Dense, Activation, Cropping1D, Lambda, Input
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.engine.topology import Layer
import pandas as pd
from src import sal

ds = 'casd'
print( '--- Loading data sets: ', ds)

gtchords     = loadset('../data/'+ds+'/allgtsuperchords.npy')[::1]
allcqt       = loadset('../data/'+ds+'/allsupercqt.npy')[::1]
allchords    = loadset('../data/'+ds+'/allsuperchords.npy')[:,::1]
fallchords   = fixall_annotators(allchords)
fgtallchords = fixall_gt(gtchords)
testpairs    = createtestpairs(allchords)

print( '--- Loading SHIP data sets')
hchords   = create_hchords_annotators(fallchords)
gthchords = create_hchords_gt(gtchords)
vocabs    = create_annotators_vocabs(fallchords, ship=False)
hvocabs   = create_annotators_vocabs(fallchords, ship=True)
gtvocab   = create_gt_vocabs(fgtallchords, ship=False)
gthvocab  = create_gt_vocabs(fgtallchords, ship=True)

print( '--- Randomizing data sets')
trainix, testix, evalix = makerandomsets([.65, .25, .10], allcqt, random_state=24)
x_train = np.hstack((allcqt[trainix],trainix[np.newaxis].T))
y_train = hchords[trainix]
gty_train = gthchords[trainix]
x_test = np.hstack((allcqt[testix],testix[np.newaxis].T))
y_test = hchords[testix]
x_eval = np.hstack((allcqt[evalix],evalix[np.newaxis].T))
y_eval = hchords[evalix]
gty_eval = gthchords[evalix]

batch_size = 128

print('--- Building models')
annmodel = makemodel(insize=(x_train.shape[1],), outsize=hchords.shape[1])
gtmodel  = makemodel(insize=(x_train.shape[1],), outsize=hchords.shape[1])
annmodel.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
gtmodel.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

print('-- Training')
epochs = 30
# build dictionary of results
trainresults = []
res = [[] for i in np.arange(len(evals))]
if trainresults == []:
    trainresults =  dict(zip(models,[dict(zip(evals,res)),dict(zip(evals,res))]))
for i in np.arange(epochs):
    print( 'epoch ' + str(i))
    annhistory = annmodel.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    # multiple source predictions
    annpreds = annmodel.predict(x_eval)
    gthistory = gtmodel.fit(x_train, gty_train, batch_size=batch_size, epochs=1, verbose=1)
    # ground truth predictions
    gtpreds = gtmodel.predict(x_eval)
    # get topchords for annotators from CASD model
    anntopchords = getmaxchords(hvocabs, vocabs, annpreds)
    # get topchords for annotators from GT model
    gttopchords = getmaxchords(hvocabs, vocabs, gtpreds)
    # get topchords for GT from CASD model
    gt_anntopchords = getmaxchords(gthvocab[:-1], gtvocab[:-1], annpreds)
    # get topchords for GT from GT model
    gt_gttopchords = getmaxchords(gthvocab[:-1], gtvocab[:-1], gtpreds)
    annboth = np.hstack((anntopchords, gt_anntopchords[np.newaxis].T))
    gtboth = np.hstack((gttopchords, gt_gttopchords[np.newaxis].T))
    for (t,e) in zip(tests,evals):
        anneval, gteval = mirexaccmul_t(annboth, evalix, evaldict[e]), mirexaccmul_t(gtboth, evalix, evaldict[e])
        trainresults['SHIP'][e] = trainresults['SHIP'][e] + [anneval]
        trainresults['ISO'][e]  = trainresults['ISO'][e] + [gteval]
        print( e + ' eval - SHIP: ' + str(anneval) + ' ISO: ' + str(gteval))


print('-- Testing')
# build dictionary of results
evals = trainresults['SHIP'].keys()
res = [0 for i in np.arange(len(evals))]
testresults =  dict(zip(models,[dict(zip(evals,res)),dict(zip(evals,res))]))
# multiple source predictions
annpreds = annmodel.predict(x_test, verbose=1)
# ground truth predictions
gtpreds = gtmodel.predict(x_test, verbose=1)
# get topchords for annotators from CASD model
anntopchords = getmaxchords(hvocabs, vocabs, annpreds)
# get topchords for annotators from GT model
gttopchords = getmaxchords(hvocabs, vocabs, gtpreds)
# get topchords for GT from CASD model
gt_anntopchords = getmaxchords(gthvocab[:-1], gtvocab[:-1], annpreds)
# get topchords for GT from GT model
gt_gttopchords = getmaxchords(gthvocab[:-1], gtvocab[:-1], gtpreds)
for (t,e) in zip(tests,evals):
    anntest = mirexaccmul_t(anntopchords, testix, evaldict[e])
    gttest = mirexaccmul_t(gttopchords, testix, evaldict[e])
    gt_anntest = mirexaccmul_t(gt_anntopchords, testix, evaldict[e])
    gt_gttest = mirexaccmul_t(gt_gttopchords, testix, evaldict[e])
    testresults['SHIP'][e] = np.concatenate((anntest, [gt_anntest]))
    testresults['ISO'][e] = np.concatenate((gttest, [gt_gttest]))

print('-- Plotting')
tabulatetestresults(testresults, tablefmt="latex")
plotres(trainresults, testresults)

print('-- Done')


