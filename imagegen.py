#!/usr/bin/python

"""Creates some random images.

Usage:
    imagegen.py -W w -H h -S s -L l OUTPUT
    imagegen.py --help

Options:
    -W w        Width in blocks
    -H h        Height in blocks
    -S s        Scale (px per block)
    -L l        Number of levels.
    --help      Show this screen

(C) 2013 Henry de Valence <hdevalence@hdevalence.ca>
Licenced under the MIT licence.
"""

import numpy as np
from PIL import Image
from docopt import docopt
from itertools import product
from scipy.stats import threshold
import random

directions = [(-1, 1)
             ,( 0, 1)
             ,( 1, 1)
             ,( 1, 0)
             ,( 1,-1)
             ,( 0,-1)
             ,(-1,-1)
             ,(-1, 0)
             ]

def walk(x,y,d,l,L,im):
    W,H = im.shape
    if random.random() < 0.1:
        random.shuffle(directions)
    for i in range(8):
        dx,dy = directions[(d+i)%8]
        x2 = (x+dx)%W
        y2 = (y+dy)%H
        if im[x2,y2] != -1:
            continue
        else:
            im[x2,y2] = l
            return walk(x2, y2, (d+i)%8, (l+1)%L, L, im)
    return

def allset(im):
    return np.where(im==-1)[0].size == 0

def gen_blocks(W,H,L):
    im = -np.ones((W,H), dtype=np.int)
    while not allset(im):
        unset = np.where(im==-1)
        x = unset[0][0]
        y = unset[1][0]
        l = random.randrange(L)
        d = random.randrange(8)
        im[x,y] = l
        walk(x,y,d,(l+1)%L,L,im)
    return im

def thresh(im,thresh):
    tlow = threshold(im,threshmin=thresh,newval=0)
    thigh = threshold(tlow,threshmax=thresh,newval=255)
    return thigh.astype(np.uint8)

def fillblock(shape,l,L):
    prob = float(l+1)/(L+1)
    return thresh(np.random.sample(shape),prob)

def expandimage(small,L,S):
    smW,smH = small.shape
    W,H = S*smW, S*smH
    im = np.zeros((W,H),np.uint8)
    xints = zip(zip(range(0,W,S),range(S,W+S,S)),range(W))
    yints = zip(zip(range(0,H,S),range(S,H+S,S)),range(H))
    blocks = product(xints,yints)
    for ((a,b),x),((c,d),y) in blocks:
        im[a:b,c:d] = fillblock((S,S), small[x,y], L)
    return im

if __name__ == "__main__":
    arguments = docopt(__doc__)
    W = int(arguments['-W'])
    H = int(arguments['-H'])
    S = int(arguments['-S'])
    L = int(arguments['-L'])
    outname = arguments['OUTPUT']
    blocks = gen_blocks(W,H,L)
    expanded = expandimage(blocks,L,S)
    output = Image.fromarray(expanded)
    output.save(outname)

