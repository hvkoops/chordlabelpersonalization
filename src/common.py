import numpy as np
import mir_eval
# import h5py
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

# fix 7ths labels
def fixlabel(l):
    r, e = l.split(':')
    if 'majmaj7' in l:
        l = r+':7'
        return l
    if 'majmin7' in l:
        l = r+':maj(b7)'
        return l
    if 'minmin7' in l:
        l = r+':min(b7)'
        return l
    else:
        return l

def ixtolabel(root,majmin,sevs,quals,exts):
    r = rs[root]
    if r == 'N':
        return r
    else:
        mm = quals[majmin]
        sev = exts[sevs]
        lab = fixlabel(r+":"+((mm+sev).replace('N','')))
        return (lab)

# Definitions of all possible notes and chord labels
ns = [(n+'#') for n in np.array(list(string.ascii_uppercase[0:7]))[[0,2,3,5,6]]]
n = list(string.ascii_uppercase[0:7])
nn = sorted(n+ns)
qualities = ['maj', 'min']
mmch = np.concatenate((['N'],[(n+':'+q) for q in qualities for n in nn]))
mmchr = np.array([np.roll(mir_eval.chord.encode(m)[1], mir_eval.chord.encode(m)[0]) for m in mmch])
mmchrt = [tuple(np.roll(mir_eval.chord.encode(m)[1], mir_eval.chord.encode(m)[0])) for m in mmch]
chdict = dict(zip(mmchrt, mmch))

# root notes + N
rs = np.concatenate((np.roll(nn,-3), ['N']))
quals = np.array(['maj','min','N'])
qualsb = np.array([[0, 1],[1, 0],[0, 0]])
exts = np.array(['maj7', 'min7', 'N'])
extsb = np.array([[0, 1],[1, 0],[0, 0]])
labels = np.concatenate((rs,quals,exts))
ixs = np.array([(r,m,s) for r in np.arange(rs.shape[0]) for m in np.arange(quals.shape[0]) for s in np.arange(exts.shape[0])])
mmixs = np.array([(r,m,s) for r in np.arange(rs.shape[0]) for m in np.arange(quals.shape[0]) for s in [2]])
chords = np.array([ixtolabel(r,m,s,quals,exts) for (r,m,s) in ixs])
mmchords = np.array([ixtolabel(r,m,s,quals,exts) for (r,m,s) in mmixs])

# dictionary of indices for intervals in chroma
chixdic = dict(zip([3,5,6,7], [(3,5), (6,8), (9,10), (10,12)]))
# major/perfect, minor/diminished, both, or none/absent
qualsmm = np.array([[0, 1],[1, 0],[1, 1],[0, 0]])
# perfect, none/absent
qualsp = np.array([[1],[0]])

# all mir_eval tests to evaluate during training and testing
tests = [mir_eval.chord.root, mir_eval.chord.majmin, mir_eval.chord.majmin_inv,
         mir_eval.chord.mirex, mir_eval.chord.thirds, mir_eval.chord.thirds_inv, 
         mir_eval.chord.triads, mir_eval.chord.triads_inv, 
         mir_eval.chord.tetrads, mir_eval.chord.tetrads_inv, 
         mir_eval.chord.sevenths, mir_eval.chord.sevenths_inv]
models = ['SHIP','ISO']
evals = ["root", "majmin", "majmin_inv", "mirex", "thirds", "thirds_inv", "triads", 
         "triads_inv", "tetrads", "tetrads_inv", "sevenths", "sevenths_inv"]
evaldict = dict(zip(evals, tests))

# Define the dense model
def makemodel(insize, outsize):
    # remove last bin for song identification
    mask = np.concatenate((np.ones(insize[0]-1),np.zeros(1)))
    a = Input(insize)
    b = Lambda(lambda x: x *mask)(a)
    c = Dense(512)(b)
    d = Dropout(0.5)(c)
    e = LeakyReLU(alpha=0.3)(d)
    f = Dense(512)(e)
    g = Dropout(0.5)(f)
    h = LeakyReLU(alpha=0.3)(g)
    i = Dense(512)(h)
    j = Dropout(0.5)(i)
    k = LeakyReLU(alpha=0.3)(j)
    l = Dense(outsize)(k)
    m = Activation('softmax')(l)
    model = Model(inputs=a, outputs=m)
    return(model)

def getmaxchords(hvocabs, vocabs, ships):
    ix = 0
    if len(hvocabs.shape) > 1: 
        mxch = np.empty(ships.shape[0], dtype=vocabs.dtype)
        for s,i in zip(ships, np.arange(ships.shape[0])):
            print( '-- computing maximum likely chords ' + '{0}\r'.format((ix)/float(ships.shape[0])),)
            mxch[i] = maxchords(s, hvocabs, vocabs)
            ix = ix + 1
        return(mxch)
    else:        
        mxch = np.empty((ships.shape[0], vocabs.shape[0]), dtype=vocabs.dtype)
        for s,i in zip(ships, np.arange(ships.shape[0])):
            for (hv,v,j) in zip(hvocabs,vocabs,np.arange(vocabs.shape[0])):
                print( '-- computing maximum likely chords ' + '{0}\r'.format((ix)/float(ships.shape[0]*vocabs.shape[0])),)
                mxch[i][j] = maxchords(s, hv, v)
                ix = ix + 1  
        return(mxch)

# make SHIP2 from a label
def labeltoSHIP2(lab):
    r, ch, b = mir_eval.chord.encode(lab)
    roots = np.eye(13)[r]
    ints  = ch
    if (b+r) == -2:
        br = 12
    if (b+r) != -2:
        br = b+r
    bass = np.eye(13)[np.mod(br, 13)]
    return np.concatenate((roots, ints, bass))

def labeltoSHIP3(lab):
    r, ch, b = mir_eval.chord.encode(lab)
    roots = np.eye(13)[r]
    ints  = []
    intervals = [1, 3, 5, 7, 8, 10]
    for i in intervals:
        if i == 7:
            ints.append(np.array(np.all(ch[i:i+1] == qualsp, axis=1), dtype=int))
        else:
            ints.append(np.array(np.all(ch[i:i+2] == qualsmm, axis=1), dtype=int))
    if (b+r) == -2:
        br = 12
    if (b+r) != -2:
        br = b+r
    bass = np.eye(13)[np.mod(br, 13)]
    return np.concatenate((roots, np.concatenate(ints), bass))


# Test labels
def testlabsinv(outlab, outgt, verbose=True):
    tests = [mir_eval.chord.root, mir_eval.chord.majmin,
                mir_eval.chord.majmin_inv,
                mir_eval.chord.mirex,
                mir_eval.chord.thirds,
                mir_eval.chord.thirds_inv,
                mir_eval.chord.triads,
                mir_eval.chord.triads_inv,
                mir_eval.chord.tetrads,
                mir_eval.chord.tetrads_inv,
                mir_eval.chord.sevenths,
                mir_eval.chord.sevenths_inv]
    testscores = np.zeros(len(tests))
    for t,i in zip(tests, np.arange(len(tests))):
        testscores[i] = scoreit(t, outlab, outgt)
    if verbose==True:
        print ("root " + str(testscores[0]))
        print ("majmin " + str(testscores[1]))
        print ("majmin inv " + str(testscores[2]))
        print ("mirex " + str(testscores[3]))
        print ("thirds " + str(testscores[4]))
        print ("thirds inv " + str(testscores[5]))
        print ("triads " + str(testscores[6]))
        print ("triads inv " + str(testscores[7]))
        print ("tetrads " + str(testscores[8]))
        print ("tetrads inv " + str(testscores[9]))
        print ("sevenths " + str(testscores[10]))
        print ("sevenths inv " + str(testscores[11]))
    return (testscores)


def labeltochroma(lab):
    lab = fixgt(lab)
    rot, chn, b = mir_eval.chord.encode(lab)
    return np.roll(chn,rot)    


def fixgt(lab):
    # if '/' in lab:
    #     lab = string.split(lab, '/')[0]
    if lab[0] == 'N':
        lab = 'N'
    if lab == 'Emin':
        lab = 'E:min'
    if lab == 'd:maj':
        lab = 'D:maj'
    if lab == 'b:maj':
        lab = 'B:maj'
    if ':6' in lab:
        lab = string.replace(lab,':6',':maj6')
    # wrong:
    if lab == 'B:7#9':
        lab = 'B:7' 
    if lab != 'N':
        root, qual, sets, bass = mir_eval.chord.split(lab)
        if (not RepresentsInt(bass)):
            lab = mir_eval.chord.join(root, qual, sets, np.abs(mir_eval.chord.scale_degree_to_bitmap('b6').argmax()-mir_eval.chord.pitch_class_to_semitone(root)))
            return lab
    l = lab.split(':')
    if len(l) > 1 and l[1] == '7sus4':
        lab = string.join([l[0],'sus4(7)'],':')
    return lab


def tabulatetestresults(testresults, tablefmt="simple", ds='casd'):
    headers=['A1','A2','A3','A4','A5','mean','A1','A2','A3','A4','A5','mean']
    if ds=='casd':
        headers=['A1','A2','A3','A4','mean','BB','A1','A2','A3','A4','mean','BB']
    res = np.array([np.concatenate((testresults['SHIP'][k],testresults['ISO'][k])) for k in testresults['SHIP']])
    if ds=='casd':
        nanns = 4
    annres = res[:,0:nanns]
    annmean = np.mean(annres, axis=1)[np.newaxis].T
    anngt = res[:,nanns:nanns+1]
    gtres = res[:,(nanns+1):nanns+(nanns+1)]
    gtmean = np.mean(gtres, axis=1)[np.newaxis].T
    gtgt = res[:,-1][np.newaxis].T
    restab = np.hstack((annres, annmean, anngt, gtres, gtmean, gtgt))
    tests = testresults['SHIP'].keys()
    allmean = np.round(restab.mean(axis=0),2)
    htab = np.hstack((np.array(tests)[np.newaxis].T,np.round(restab,2)))
    hmtab = np.vstack((htab, np.concatenate((['mean'], allmean))))
    print(tabulate(hmtab, headers=headers, tablefmt=tablefmt))

def heatmaptestresults(testresults, tablefmt="simple", ds='casd'):
    headers=['A1','A2','A3','A4','A5','mean','A1','A2','A3','A4','A5','mean']
    if ds=='casd':
        headers=['A1','A2','A3','A4','mean','A1','A2','A3','A4','mean']
    res = np.array([np.concatenate((testresults['SHIP'][k],testresults['ISO'][k])) for k in testresults['SHIP']])
    nanns = len(testresults['SHIP']['root'])
    restab = np.hstack((res[:,0:nanns], np.mean(res[:,0:nanns], axis=1)[np.newaxis].T, res[:,nanns:], np.mean(res[:,nanns:], axis=1)[np.newaxis].T))
    tests = testresults['SHIP'].keys()
    allmean = np.round(restab.mean(axis=0),2)
    htab = np.hstack((np.array(tests)[np.newaxis].T,np.round(restab,2)))
    hmtab = np.vstack((htab, np.concatenate((['mean'], allmean))))
    print(tabulate(hmtab, headers=headers, tablefmt=tablefmt))


def savecqtfeatures(wavpath, ds='ni'):
    y, sr = librosa.load(wavpath, sr=44100)
    cqt = np.abs(librosa.core.cqt(y, sr=sr, hop_length=4096, fmin=librosa.note_to_hz('C1'), n_bins=24 * 8, bins_per_octave=24, real=False)).astype(np.float32)
    npspec = librosa.util.normalize(cqt, norm=np.inf, axis=1).T
    featpath = 'cqt/' + ds + '/' + wavpath.split('/')[-1].split('.')[0] + '.npy'
    np.save(featpath, npspec)

def savescores(scores, date, dataset):
    np.save('scores_'+dataset+'_'+date+'.npy', scores)

def makerandomsets(sizes, frames, random_state=24):
    trainsize = sizes[0]
    testsize = sizes[1]
    evalsize = ((trainsize+sizes[2])*sizes[2])
    dummyframes = np.arange(frames.shape[0])
    dummychords = np.arange(frames.shape[0])
    dummygt = np.arange(frames.shape[0])
    X_trainix, X_testix, y_trainix, y_testix = model_selection.train_test_split(dummyframes, dummychords, test_size=1-sizes[0], random_state=random_state)
    X_trainix, X_evalix, y_trainix, y_evalix = model_selection.train_test_split(X_trainix, y_trainix, test_size=evalsize, random_state=random_state)
    return (X_trainix, X_testix, X_evalix)

def makesuperframes(frames):
    superframesize = 15
    hsuperframe = (superframesize/2)
    featureshape = 192 
    nsuperframes = featureshape*superframesize    
    n_superframes = frames.shape[0] - (superframesize-1)
    superframes = np.array([frames[s:s+superframesize].flatten() for s in np.arange(n_superframes)])
    return superframes

def makedatasets(ds='ni'):
    songs = np.array([s.split('/')[-1][:-4] for s in glob('audio/'+ds+'/*.wav')])
    allcqt = makesuperframes(np.load('cqt/'+ds+'/' + songs[0] + '.npy'))
    allchords = np.load('chords/'+ds+'/anns/' + songs[0] + '.npy')[:,0:allcqt.shape[0]]
    gtchords = np.load('chords/'+ds+'/gt/' + songs[0] + '.npy')[0:allcqt.shape[0]]
    for s in songs[1:]:
        print(s)
        superframes = makesuperframes(np.load('cqt/'+ds+'/' + s + '.npy'))
        allchords = np.hstack((allchords, np.load('chords/'+ds+'/anns/' + s + '.npy')[:,0:superframes.shape[0]]))
        gtchords = np.concatenate((gtchords, np.load('chords/'+ds+'/gt/' + s + '.npy')[0:superframes.shape[0]]))
        allcqt = np.vstack((allcqt, superframes))
    np.save('data/'+ds+'/allsuperchords.npy', allchords)
    np.save('data/'+ds+'/allsupercqt.npy', allcqt)
    np.save('data/'+ds+'/allgtsuperchords.npy', gtchords)

def makegtdatasets():
    songs = np.array(glob('GT/*'))
    allcqt = makesuperframes(np.load('cqt/' + songs[0].split('/')[1]))
    gtchords = np.load(songs[0])[0:allcqt.shape[0]]
    for s in songs[1:]:
        superframes = makesuperframes(np.load('cqt/' + s.split('/')[1]))
        gtchords = np.concatenate((gtchords, np.load('gt/' + s.split('/')[1])[0:superframes.shape[0]]))
    np.save('allgtsuperchords.npy', gtchords)

def hierachytolabel(hir):
    r = rs[hir[0:13].argmax()]
    if r == 'N':
        return r
    else:
        mm = quals[hir[13:16].argmax()]
        sev = exts[hir[16:].argmax()]
        lab = fixlabel(r+":"+((mm+sev).replace('N','')))
        return (lab)

def hierachytolabel_mul(hir):
    probs = np.array([r*m*s for r in hir[0:13] for m in hir[13:16] for s in hir[16:]])
    return(chords, probs)

def hierachytolabel_mul(hir, level=0):
    if level==2:
        probs = np.array([r*m*s for r in hir[0:13] for m in hir[13:16] for s in hir[16:]])
    return(chords, probs)

def hierachytolabelprobs(hir):
    probs = np.array([r*m*s for r in hir[0:13] for m in hir[13:16] for s in hir[16:]])
    return(probs)

def gttoscore(gts):
    bc = np.bincount(np.array([np.where(chords == g)[0][0] for g in gts]))
    if bc.shape != chords.shape[0]:
        bc = np.concatenate((bc, np.zeros(chords.shape[0]-bc.shape[0])))
    return (bc/float(gts.shape[0]))        


def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# compute a likelihood for a dnn output given a known SHIP 
def vs(dnnship, gtship):
    epsilon = np.finfo(np.float64).eps
    ps = ((dnnship + epsilon) * gtship)
    ps0 = ps[ps > 0]
    prob = np.product(ps0)
    return (prob)

# rank the chords of a vocabulary according to hchord (ship)
def rankchords(ship, vocab):
    rank = np.array([vs(ship, labeltohierarchy(v)) for v in vocab])
    return vocab[np.argsort(rank)[-1]]

def maxchords(ship, hvocab, vocab):
    rank = np.array([vs(ship, v) for v in hvocab])
    return vocab[np.argmax(rank)]

def mirexaccmul_t(topchords, cids, test):
    if len(topchords.shape) > 1:
        anngt = np.vstack((fallchords[:,cids], fgtallchords[cids]))
        scores = np.array([scoreit(test, f, t) for (f,t) in zip(anngt, topchords.T)])
    else:
        scores = scoreit(test, fgtallchords[cids], topchords)
    return scores

def scoreit(func, outlab, outgt):
    res = func(outlab, outgt)
    score = res[res >= 0].sum()/float(res[res >= 0].shape[0])
    return score    