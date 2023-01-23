#!/usr/bin/env python

import os
import sys
import argparse
import pdb
import time

# Internal imports
from alicesaur.pipeline.pipeline import Pipeline


if __name__ == "__main__":

    desc = 'Reduce STIS imaging data'
    parser = argparse.ArgumentParser(description=desc)

    desc = 'Path to directory containing FITS data for processing.'
    parser.add_argument('--dataDir', metavar='dataDir',
                        default="./", type=str, nargs='?',
                        help=desc)

    desc = 'Observation mode: bar10 or wedgeb1.0'
    parser.add_argument('--obsMode', metavar='obsMode',
                        default="bar10", type=str, nargs='?',
                        help=desc)

    desc = 'PSF subtraction mode: rdi, adi, pyklip-rdi, or pyklip-adi'
    parser.add_argument('--psfSubMode', metavar='psfSubMode',
                        default="rdi", type=str, nargs='?',
                        help=desc)

    desc = 'Target name (e.g., HD-32297)'
    parser.add_argument('--targ', metavar='targ',
                        default="", type=str, nargs='?',
                        help=desc)

    desc = 'Background sampling patch center y x (e.g., 735 627)'
    parser.add_argument('--bgCen', metavar='bgCen',
                        default=None, type=str, nargs='*',
                        help=desc)

    desc = 'Number of annuli to use for subtraction optimization (default 1)'
    parser.add_argument('--ann', metavar='ann',
                        default=1, type=int, nargs='?',
                        help=desc)

    desc = 'Width in [pixels] of diffraction spike masks. Default is to use the value in the info.json.'
    parser.add_argument('--spWidth', metavar='spWidth',
                        default=None, type=int, nargs='?',
                        help=desc)

    desc = 'Flag to save the final image to FITS, overwriting any existing file.'
    parser.add_argument('--saveFinal', dest="saveFinal",
                        action="store_true", help=desc)

    desc = 'Use --noFixPix to NOT fix bad pixels in each individual image.'
    parser.add_argument('--noFixPix', dest="noFixPix",
                        action="store_true", help=desc)

    desc = 'Use --noRadon to NOT do radon star centering on each image.'
    parser.add_argument('--noRadon', dest="noRadon",
                        action="store_true", help=desc)

    desc = 'Use --noCombine to NOT combine images from each orbit before processing.'
    parser.add_argument('--noCombine', dest="noCombine",
                        action="store_true", help=desc)

    desc = 'Use --noPad to NOT pad images into 2048x2048 arrays.'
    parser.add_argument('--noPad', dest="noPad",
                        action="store_true", help=desc)

    desc = 'Use --noErrorMaps to NOT make error maps or SNR maps.'
    parser.add_argument('--noErrorMaps', dest="noErrorMaps",
                        action="store_true", help=desc)

    args = parser.parse_args()

    # Create the reduction pipeline instance.
    pl = Pipeline(**vars(args))

    # Run the reduction.
    pl.run()
