#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:19:14 2023

@author: tom
"""
import os
import shutil
from distutils import spawn
import numpy as np
from astropy.io import fits
# from drizzlepac import astrodrizzle
# from drizzlepac.stisData import STISInputImage
from stistools.x2d import x2d


def correct_distortion(imgPathList, outputPaths="", refDir=None, inst='stis',
                       overwrite=False):

    if type(outputPaths) is str:
        if os.path.isdir(outputPaths):
            outputPaths = len(imgPathList)*[outputPaths]
        else:
            print("\nWARNING: outputPaths for correct_distortion can be only"\
                  " a list of strings or a single str path to a directory."\
                  " Forcing outputs to add a _x2d suffix to input image "\
                  "paths.\n")
            outputPaths = ""
    if outputPaths == "":
        outputPaths = len(imgPathList)*[""]

    # Set environment path to IDC table file, if given manually.
    if refDir is not None:
        os.environ['oref'] = os.path.abspath(refDir)
    else:
        if os.environ.get('oref') is None:
            # Must have trailing slash in 'oref' path.
            default_oref_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/distortion')), '')
            os.environ['oref'] = default_oref_path
            print("\nWARNING: Environment variable 'oref' was not defined"\
                  " so distortion correction may fail. "\
                  f"Resetting 'oref' to {default_oref_path}.\nYou may need"\
                  " to set it manually to the path to the IDC table file's"\
                  " parent directory or provide that path here with the"\
                  " refDir argument.")

    success_list = undistort_images(imgPathList, outputPaths=outputPaths,
                                    overwrite=overwrite)

    return success_list


def undistort_images(imgPathList, outputPaths="", overwrite=False):

    if overwrite:
        print("WARNING: Will overwrite existing x2d.fits files")
        
    success_list = []

    for ii, imgPath in enumerate(imgPathList):
        success_list.append(undistort_single_image(imgPath, outputPath=outputPaths[ii],
                                         overwrite=overwrite))

    return np.array(success_list)


def undistort_single_image(imgPath, outputPath="", overwrite=False):
    """
    Correct geometric distortion of a single STIS image.
    
    ***
    TOM'S NOTES ON X2D: The stistools.x2d.x2d function is almost purely
    a wrapper for calling the cs7.e executable. So we can easily replace our
    dependence on x2d with a (simpler) wrapper of our own.
    ***

    Parameters
    ----------
    imgPath : TYPE
        DESCRIPTION.
    outputPath : TYPE, optional
        DESCRIPTION. The default is "".
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None. Writes a new *_x2d.fits file with the undistorted image(s).

    """

    print(os.environ.get('oref'))

    if os.environ.get('oref') is None:
        print('\nWARNING: Environment variable "oref" is not defined'\
              ' so distortion correction will likely fail. You may need'\
              ' to set it manually as the path to the IDC table file'\
              ' or provide that path here with the idcPath argument.\n')

    # Check for presence of required IRAF executable named 'cs7.e'.
    path_cs7e = spawn.find_executable('cs7.e')
    if path_cs7e is None:
        print('\nWARNING: CALSTIS executable cs7.e not found in your environment.'\
              ' Distortion correction will not work without it.')
    # If cs7.e is found, make sure it is in the environment path and add it if
    # it is not.
    else:
        if os.path.dirname(path_cs7e) not in os.environ['PATH']:
            print('\ncs7.e CALSTIS executable was not in the environment '\
                  'PATH. Adding it.')
            os.environ["PATH"] += os.pathsep + os.path.dirname(path_cs7e)
        else:
            pass
        
    # If overwriting existing files, keep a backup copy until the correction
    # has finished successfully.
    if overwrite:
        if os.path.exists(outputPath):
            try:
                shutil.move(outputPath, outputPath + '-backup')
            except:
                print("\nWARNING: Could not back up existing "\
                      f"distortion-corrected x2d.fits at {outputPath}\n")

    # Do the distortion correction.
    x2d_status = x2d(imgPath, output=outputPath, fluxcorr=False, helcorr=False)

    if x2d_status == 0:
        if overwrite:
            if os.path.exists(outputPath + '-backup'):
                os.remove(outputPath + '-backup')
    elif x2d_status == 1:
        print("\nHELP! Distortion correction failed. Either the required "\
              "IRAF executable 'cs7.e' was not found or the correction"\
              " was attempted and failed. Please make sure cs7.e is in your"\
              " environment's system path.")
    elif x2d_status == 2:
        print("\nHELP! Distortion correction failed. Either the required "\
              "IRAF executable 'cs7.e' was not found or the correction"\
              " was attempted and failed. Please make sure cs7.e is in your"\
              " environment's system path.")
    elif x2d_status > 2:
        print("\nWARNING! Something unknown went wrong during distortion"\
              "correction.\n")

    # For subarrayed images, remove the zero-padding from top & right edges of
    # images created by cs7.e. Otherwise, we get strips of zeros in our
    # derotated cubes. It seems to be a constant 38 pixels on all edges.
    # Bottom and left edges need to be kept to match the occulter mask.
    padY = 38
    padX = 38
    with fits.open(outputPath, mode='update') as hdul:
        shape = hdul[1].data.shape
        if (shape[0] < 1100) | (shape[1] < 1100):
            for ii in range(1, len(hdul)):
                shape = hdul[ii].data.shape
                hdul[ii].data = hdul[ii].data[:shape[0] - padY,
                                              :shape[1] - padX]
        print("Subarrayed image detected: trimmed zero-padding off of the "\
              f" distortion-corrected arrays top edge (y={padY},x={padX} pix).")

    # Delete backup or move backup back to original depending on success.
    if x2d_status == 0:
        if overwrite:
            if os.path.exists(outputPath + '-backup'):
                os.remove(outputPath + '-backup')

        return True

    else:
        if overwrite:
            if os.path.exists(outputPath + '-backup'):
                shutil.move(outputPath + '-backup', outputPath)

        return False


# def custom_x2d_cs7(fileList):
    
#     for (i, infile) in enumerate(infiles):

#         arglist = ["cs7.e"]

#         arglist.append(infile)
#         if outfiles is not None:
#             arglist.append(outfiles[i])

#         if verbose:
#             arglist.append("-v")
#         if timestamps:
#             arglist.append("-t")

#         switch_was_set = False
#         if helcorr == "perform":
#             arglist.append("-hel")
#             switch_was_set = True
#         if fluxcorr == "perform":
#             arglist.append("-flux")
#             switch_was_set = True
#         if not switch_was_set:
#             arglist.append("-x2d")

#         if err_alg:
#             if err_alg == "wgt_err":
#                 arglist.append("-wgt_err")
#             elif err_alg != "wgt_var":
#                 raise RuntimeError("err_alg must be either 'wgt_err'"
#                                    " or 'wgt_var'; you specified '%s'" % err_alg)

#         if blazeshift is not None:
#             arglist.append("-b")
#             arglist.append("%.10g" % blazeshift)

#         if verbose:
#             print("Running x2d on {}".format(infile))
#             print("  {}".format(arglist))
#         status = subprocess.call(arglist, stdout=fd_trailer,
#                                  stderr=subprocess.STDOUT)
#         if status:
#             cumulative_status = 1
#             if verbose:
#                 print("Warning:  status = {}".format(status))
    
    