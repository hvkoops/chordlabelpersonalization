import numpy as np
import mir_eval
import h5py
import tensorflow as tf
from sklearn import cross_validation
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

# sys.setdefaultencoding('utf8')
def plotships3():
    fig = plt.figure(figsize=(15,1.5))
    ax = fig.add_subplot(111)
    labels = np.array([u'G:min', u'Eb:maj/3', u'G:min', u'Eb:maj/3', u'Eb:maj'], dtype='<U20')
    ships = np.array([labeltoSHIP3(c) for c in labels])
    withmean = np.vstack((ships, ships.mean(axis=0)))
    meanlabels = np.concatenate(([''], labels, ['SHIP']))
    chromalabels = np.concatenate((np.roll(nn,-3), ['N']))
    # major/perfect, minor/diminished, both, or none/absent
    intlabels = ['2', '♭2', '2B', '*2', '3', '♭3', '3B', '*3', '4', '#4', '4B', '*4', '5', '*5', '6', '♭6', '6B', '*6', '7', '♭7', '7B', '*7']
    shiplabels = np.concatenate((chromalabels, intlabels, chromalabels))
    ax.matshow(withmean, cmap=plt.cm.Greys, aspect='auto')
    ax.axvline(x=12.5, ymax=1.105, color='k', clip_on=False, linewidth=2.0)
    ax.axvline(x=34.5, ymax=1.105, color='k', clip_on=False, linewidth=2.0)
    ax.set_yticklabels(meanlabels)
    ax.set_xticks(np.arange(ships.shape[1]))
    ax.set_xticklabels(shiplabels)
    for i in np.arange(withmean.shape[1]):
        for j in np.arange(withmean.shape[0]):
            if (withmean[j][i] == 1):
                ax.annotate(int(withmean[j][i]), xy=(i, j+0.2), color='w', fontsize=10, ha='center')
            if (withmean[j][i] == 0):
                ax.annotate(int(withmean[j][i]), xy=(i, j+0.2), fontsize=10, ha='center')
            if (withmean[j][i] < 1) and (withmean[j][i] > 0):
                ax.annotate('.' + str(withmean[j][i]).split('.')[1], xy=(i, j+0.2), fontsize=10, ha='center')
    for i in np.arange(-1,ships.shape[1]):
        ax.axvline(x=i+0.5, linestyle='dashed', color='k', ymax=1.1, clip_on=False, linewidth=0.5)
    for j in np.arange(ships.shape[0]):
        if (j == ships.shape[0]-1):
            ax.axhline(y=j+0.5, linestyle='dashed', xmin=-0.075, clip_on=False, color='k', linewidth=2.0)
        else:
            ax.axhline(y=j+0.5, linestyle='dashed', color='k', linewidth=0.5)
    plt.savefig('plots/hipships2.pdf', bbox_inches='tight')
    plt.close()

def plotships():
    fig = plt.figure(figsize=(13,1.5))
    ax = fig.add_subplot(111)
    labels = np.array(['G:maj7', 'G:maj', 'G:maj7', 'G:maj/3'])
    ships = np.array([labeltoSHIP3(c) for c in labels])
    withmean = np.vstack((ships, ships.mean(axis=0)))
    meanlabels = np.concatenate(([''], labels, ['SHIP']))
    chromalabels = np.concatenate((np.roll(nn,-3), ['N']))
    # [1, 3, 5, 7, 8, 10]
    intlabels = ['*2', 'b2', '2', '2B', '*3', 'b3', '3', '3B', '*4', 'b4', '4', '4B', '*5', '5', '*6', 'b6', '6', '6B', '*7', 'b7', '7', '7B']
    # shiplabels = np.concatenate((chromalabels, np.array(np.arange(1,13), dtype='S'), chromalabels))
    shiplabels = np.concatenate((chromalabels, np.array(np.arange(1,13), dtype='S'), chromalabels))
    ax.matshow(withmean, cmap=plt.cm.Greys, aspect='auto')
    ax.axvline(x=12.5, ymax=1.105, color='k', clip_on=False, linewidth=2.0)
    ax.axvline(x=24.5, ymax=1.105, color='k', clip_on=False, linewidth=2.0)
    ax.set_yticklabels(meanlabels)
    ax.set_xticks(np.arange(ships.shape[1]))
    ax.set_xticklabels(shiplabels)
    for i in np.arange(withmean.shape[1]):
        for j in np.arange(withmean.shape[0]):
            if (withmean[j][i] == 1):
                ax.annotate(int(withmean[j][i]), xy=(i, j+0.2), color='w', fontsize=10, ha='center')
            if (withmean[j][i] == 0):
                ax.annotate(int(withmean[j][i]), xy=(i, j+0.2), fontsize=10, ha='center')
            if (withmean[j][i] < 1) and (withmean[j][i] > 0):
                ax.annotate('.' + str(withmean[j][i]).split('.')[1], xy=(i, j+0.2), fontsize=10, ha='center')
    for i in np.arange(-1,ships.shape[1]):
        ax.axvline(x=i+0.5, linestyle='dashed', color='k', ymax=1.1, clip_on=False, linewidth=0.5)
    for j in np.arange(ships.shape[0]):
        if (j == ships.shape[0]-1):
            ax.axhline(y=j+0.5, linestyle='dashed', xmin=-0.075, clip_on=False, color='k', linewidth=2.0)
        else:
            ax.axhline(y=j+0.5, linestyle='dashed', color='k', linewidth=0.5)
    plt.savefig('plots/hipships.pdf', bbox_inches='tight')
    plt.close()


def plotpairwiseagreement(pairscores):
    usetests = [0,1,3,4,6,8,10]
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(111)
    cax = ax.matshow(pairscores[:,:,usetests].mean(axis=1).T, aspect='auto', cmap=plt.cm.Blues)
    ax3 = ax.twinx()
    cbaxes = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cb = plt.colorbar(cax, cax = cbaxes)
    tests = np.array(["root", "majmin", "majmin_inv", "mirex", "thirds", "thirds_inv", "triads", "triads_inv", "tetrads", "tedrads_inv", "sevenths", "sevenths_inv"])
    ax.set_yticks(np.arange(tests[usetests].shape[0]))
    ax.set_yticklabels(tests[usetests])
    ax.set_xticks(np.arange(pairscores[:,:,usetests].shape[0]))
    ax.set_xticklabels(np.arange(pairscores[:,:,usetests].shape[0])+1)
    ax.set_xlim(-0.5, pairscores[:,:,usetests].shape[0]-0.5)
    ax3.set_yticks(np.arange(pairscores[:,:,usetests].shape[2]))
    ax3.set_yticklabels(np.array([str(np.round(s,2))[0:4] for s in pairscores[:,:,usetests].mean(axis=1).T.mean(axis=1)])[::-1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('plots/pwcorr_bbq.pdf', bbox_inches='tight')
    plt.close()

def taskscoresinv(pairscores):
    usetests = [0,1,3,4,6,8,10]
    taskscores = np.array([pairscores[:,:,usetests][:,:,pa].flatten() for pa in np.arange(pairscores[:,:,usetests].shape[2])])
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('Pastel2')
    cix = 0
    tests = np.array(["root", "majmin", "majmin_inv", "mirex", "thirds", "thirds_inv", "triads", "triads_inv", "tetrads", "tedrads_inv", "sevenths", "sevenths_inv"])
    ncolors = ["black", "firebrick", "rosybrown", "darkorange", "darksage", "lightgreen", "darkviolet", "violet", "gold", "darkkhaki", "tan", "bisque", "darkcyan", "lightcyan"]
    bplots = ax.boxplot(taskscores.T, notch=True, vert=True, patch_artist=True, widths=0.7)
    colors = [cmap(cix) for cix in np.arange(0., (1.), 1/float(len(usetests)))]
    for bplot in bplots:
        for patch, median, whisker, color in zip(bplots['boxes'], bplots['medians'], np.array(bplots['whiskers']).reshape(len(usetests),2), colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('k')
            median.set_color('k')
            whisker[0].set_color('k')
            whisker[1].set_color('k')
    # caps = np.array(bplots['caps'])[1:][::2][[1,2,4,5,6,7,8,9,10,11]]        
    # cheights = np.array([k.get_ydata()[0] for k in np.array(bplots['caps'][1:][::2])[[1,2,4,5,6,7,8,9,10,11]]])
    ax.set_xticklabels(tests[usetests], rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    for label in ax.get_xmajorticklabels() + ax.get_xmajorticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    plt.ylabel('Pairwise agreement', fontsize=13)    
    plt.savefig('plots/taskscoresinvall.pdf', bbox_inches='tight')
    plt.close()    


def plothistory(history):
    fig, ax = plt.subplots()
    [ax.plot(history.history[k], label=k) for k in history.history.keys()]
    legend = ax.legend(loc='upper center')
    plt.show()

# plot the train and testresults
def plotres(trainresults, testresults):
    mmcmap = plt.cm.get_cmap('Blues')
    rcmap = plt.cm.get_cmap('Reds')
    cmaps = dict(zip(trainresults.keys(),[rcmap,mmcmap]))
    f, axarr = plt.subplots(len(trainresults['SHIP'])/2, 2, sharex=True, sharey=True, figsize=(30, 20))
    for (c,i) in zip(axarr, np.arange(len(axarr))):
        for (r,j) in zip(c, np.arange(len(c))):
            for m in models:
                tr = np.array(trainresults[m][evals[i+j]])
                axarr[i][j].errorbar(np.arange(len(trainresults['SHIP']['root'])), tr.mean(axis=1), yerr=tr.std(axis=1), label=m, color=cmaps[m](1.))
                te = np.array(testresults[m][evals[i+j]])
                for e in te:
                    axarr[i][j].plot(np.repeat(e, len(tr)), color=cmaps[m](1.))
                axarr[i][j].set_title(evals[i+j])
                axarr[i][j].set_xlabel('epoch')
                axarr[i][j].set_ylabel('accuracy')
                axarr[i][j].legend()
                axarr[i][j].set_ylim([0.4,1.])
    plt.savefig('plots/traintestresults.pdf', bbox_inches='tight')
    plt.show()    