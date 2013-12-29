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

class Directions(object):
    """
    Wrapper around a list of directions that
    can cyclically rotate their order.
    """
    _dirs = [ (-1,  1)
            , ( 0,  1)
            , ( 1,  1)
            , ( 1,  0)
            , ( 1, -1)
            , ( 0, -1)
            , (-1, -1)
            , (-1,  0)
            ]
    def __init__(self, directionality):
        self._directionality = directionality
        self._dir_offset = 0
        self.change_direction()

    def d(self, i):
        return self._dirs[(i + self._dir_offset) % 8]

    def change_direction(self):
        if random.random() < self._directionality:
            self._dir_offset = random.randrange(8)

def allset(im):
    """
    Return true if im has no -1 values.
    """
    return np.where(im==-1)[0].size == 0

def seek(x, y, im, directions):
    """
    Seeks a new location from (x, y) which is unset in im.
    """
    directions.change_direction()
    width, height = im.shape
    for i in range(8):
        dx, dy = directions.d(i)
        x2 = (x + dx) % width
        y2 = (y + dy) % height
        if im[x2, y2] == -1:
            return (x2, y2)
    return None

def gen_blocks(width, height, num_levels, directionality):
    """
    Generate an image according to a random algorithm.
    """
    im = -np.ones((width, height), dtype=np.int)
    while not allset(im):
        unset = np.where(im==-1)
        x, y = unset[0][0], unset[1][0]
        level = random.randrange(num_levels)
        dirs = Directions(directionality)
        while True:
            im[x, y] = level
            next_pos = seek(x, y, im, dirs)
            if not next_pos:
                break # Hit a dead end
            else:
                x, y = next_pos
                level = (level + 1) % num_levels
    return im.transpose() # numpy rowmajor, PIL columnmajor

def thresh(im, thresh):
    """
    Threshold image with threshdold value thresh
    """
    tlow = threshold(im, threshmin=thresh, newval=0)
    thigh = threshold(tlow, threshmax=thresh, newval=255)
    return thigh.astype(np.uint8)

def fillnoise(shape, level, max_levels):
    """
    Fills blocks with binary noise.
    """
    prob = float(level+1)/(max_levels+1)
    return thresh(np.random.sample(shape), prob)

def fillpalette(shape, level, palette):
    """
    Fills blocks according to a palette.
    palette -- array of uint8 3-tuples.
    """
    b = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    b[:, :, :] = palette[level]
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
    with open(fname, 'r') as inputfile:
        lines = inputfile.readlines()
        palette = np.zeros((len(lines), 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            palette[i] = [int(x) for x in line.split(',')]
    return palette

if __name__ == "__main__":
    arguments = docopt(__doc__)
    width = int(arguments['-W'])
    height = int(arguments['-H'])
    blocksize = int(arguments['-S'])
    directionality = float(arguments['-D'])
    outname = arguments['OUTPUT']
    if '-P' in arguments:
        # Load colour palette from file
        palette = loadpalette(arguments['-P'])
        expanded = np.zeros((blocksize*height, blocksize*width, 3),
                            dtype=np.uint8)
        def fill(shape, val):
            return fillpalette(shape, val, palette)
        num_levels = palette.shape[0]
    else:
        # black and white, use fillnoise
        num_levels = int(arguments['-L'])
        def fill(shape, val):
            return fillnoise(shape, val, num_levels)
        expanded = np.zeros((blocksize*height, blocksize*width),
                            dtype=np.uint8)
    small = gen_blocks(width, height, num_levels, directionality)
    large = expandimage(small, expanded, blocksize, fill)
    output = Image.fromarray(large)
    output.save(outname)

