#!/usr/bin/python

"""Creates some random images.

Usage:
    imagegen.py -W w -H h [-S s] [-D d] (-L l | -P palette) OUTPUT
    imagegen.py --help

Options:
    -W w        Width in blocks
    -H h        Height in blocks
    -S s        Scale, px per block [default: 32]
    -D d        Directionality: 0-1, 1 fully random [default: 0.1]
    -L l        Number of levels (greyscale noise)
    -P palette  Palette file (colour output)
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

directions = [ (-1,  1)
             , ( 0,  1)
             , ( 1,  1)
             , ( 1,  0)
             , ( 1, -1)
             , ( 0, -1)
             , (-1, -1)
             , (-1,  0)
             ]

directionality = 0.1

def walk(x, y, d, l, L, im):
    W, H = im.shape
    if random.random() < directionality:
        random.shuffle(directions)
    for i in range(8):
        dx, dy = directions[(d+i)%8]
        x2 = (x+dx)%W
        y2 = (y+dy)%H
        if im[x2, y2] != -1:
            continue
        else:
            im[x2, y2] = l
            return walk(x2, y2, (d+i)%8, (l+1)%L, L, im)
    return

def allset(im):
    """
    Return true if im has no -1 values.
    """
    return np.where(im==-1)[0].size == 0

def gen_blocks(W, H, L):
    """
    Generate an image according to a random algorithm.
    """
    im = -np.ones((W, H), dtype=np.int)
    while not allset(im):
        unset = np.where(im==-1)
        x = unset[0][0]
        y = unset[1][0]
        l = random.randrange(L)
        d = random.randrange(8)
        im[x, y] = l
        walk(x, y, d, (l+1)%L, L, im)
    return im

def thresh(im, thresh):
    """
    Threshold image with threshdold value thresh
    """
    tlow = threshold(im, threshmin=thresh, newval=0)
    thigh = threshold(tlow, threshmax=thresh, newval=255)
    return thigh.astype(np.uint8)

def fillnoise(shape, l, L):
    """
    Fills blocks with binary noise.
    L -- integer representing number of levels.
    """
    prob = float(l+1)/(L+1)
    return thresh(np.random.sample(shape), prob)

def fillpalette(shape, l, palette):
    """
    Fills blocks according to a palette.
    palette -- array of uint8 3-tuples.
    """
    h, w = shape
    b = np.zeros((h, w, 3), dtype=np.uint8)
    b[:, :, :] = palette[l]
    return b

def expandimage(small, large, S, fill):
    """
    Expands small image into coloured blocks.
    small   -- the small image
    large   -- the large image
    S       -- size of blocks.
    fill    -- function to create blocks.
    """
    H, W = S*small.shape[0], S*small.shape[1]
    xints = zip(zip(range(0, H, S), range(S, H+S, S)), range(H))
    yints = zip(zip(range(0, W, S), range(S, W+S, S)), range(W))
    blocks = product(xints, yints)
    for ((a, b), x), ((c, d), y) in blocks:
        large[a:b, c:d] = fill((S, S), small[x, y])
    return large

def loadpalette(fname):
    """
    Load a palette file.
    Format is R,G,B integers, one per line.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
        palette = np.zeros((len(lines), 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            palette[i] = list(map(int, line.split(',')))
    return palette

if __name__ == "__main__":
    arguments = docopt(__doc__)
    W = int(arguments['-W'])
    H = int(arguments['-H'])
    S = int(arguments['-S'])
    directionality = float(arguments['-D'])
    outname = arguments['OUTPUT']
    if '-P' in arguments:
        palette = loadpalette(arguments['-P'])
        expanded = np.zeros((S*H, S*W, 3), dtype=np.uint8)
        def fill(shape, val):
            return fillpalette(shape, val, palette)
        L = palette.shape[0]
    else: # black and white
        L = int(arguments['-L'])
        def fill(shape, val):
            return fillnoise(shape, val, L)
        expanded = np.zeros((S*H, S*W), dtype=np.uint8)
    small = gen_blocks(H, W, L)
    large = expandimage(small, expanded, S, fill)
    output = Image.fromarray(large)
    output.save(outname)

