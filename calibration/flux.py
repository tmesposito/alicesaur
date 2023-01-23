#!/usr/bin/env python

import sys
import pdb
import numpy as np

# Internal imports
from alicesaur import utils


def convert_intensity(imgs, hdrs, unitStart=None, unitEnd='counts s-1',
                      gain=None, exptime=None, nCombine=None, pscale=0.0506):
    """
    Convert image from one set of intensity units (aka "flux) to another).
    Possible unit strings are 'counts', 'counts s-1', 'e-', 'jy', 'mjy',
    'jy arcsec-2', or 'mjy arcsec-2'.

    Conversion from counts s-1 to Jy for gain=4.016 adopted from
    Schneider+ 20 () after discussion with P. Kalas.

    Inputs:
        imgs: list of 2-D image arrays to be converted.
        hdrs: list of headers matching imgs in length and order.
        unitStart: str, units of input image. (Default) None will use BUNIT key.
        unitEnd: str, units to output the image in. (Default is 'counts s-1')
        gain: float, detector gain in [counts/e-].
        exptime: float, constituent image exposure time in [s].
        nCombine: int, number of constituent images with exposure time exptime.
        pscale: float, pixel scale of images in [arcsec/pixel]. Defaults
            to STIS' 0.0506 arcsec/pixel.
    """

    # Redefine some variables for nicer variable names later.
    exptime_force = exptime
    gain_force = gain
    nCombine_force = nCombine

    for ii, im in enumerate(imgs):
        if unitStart is None:
            bunit = hdrs[ii][1]['BUNIT'].lower()
        else:
            bunit = unitStart.lower()

        # EXPTIME in sx2 header is total (summed) exposure time per sx2 image (NOT per
        # NCOMBINE coadded constituent frames.
        # Paul has confirmed this with his own tests.
        if nCombine_force is None:
            nCombine = hdrs[ii][1]['NCOMBINE'] # 
        else:
            nCombine = nCombine_force
        if exptime_force is None:
            exptime = hdrs[ii][1]['EXPTIME']
        else:
            exptime = exptime_force
        if gain_force is None:
            gain = hdrs[ii][0]['CCDGAIN']
            # Measured conversion for CCDGAIN=4 is 4.016 +/- 0.003 counts/e-
            # From STIS Instrument Handbook (IHB) version 9 and Schneider+ 2016.
            if gain == 4:
                gain = 4.016 # [counts/e-]
        else:
            gain = gain_force

        # First, convert units to COUNTS as a uniform starting place.
        if bunit == 'counts':
            pass
        elif bunit == 'counts s-1':
            # imgs[ii] *= (exptime/nCombine) # WRONG!!! Paul confirmed this is incorrect.
            imgs[ii] *= exptime # [counts]
            hdrs[ii][1]['BUNIT'] = 'COUNTS'
        elif bunit == 'e-':
            imgs[ii] *= gain
            hdrs[ii][1]['BUNIT'] = 'COUNTS'

        # Now do the final unit conversion.
        if unitEnd.lower() == 'counts s-1':
            # Divide by integration time per image (exptime/nCombine).
            # imgs[ii] /= (exptime/nCombine) # WRONG!!! Paul confirmed this is incorrect.
            imgs[ii] /= exptime # [counts s-1]
            hdrs[ii][1]['BUNIT'] = 'COUNTS S-1'
        elif unitEnd.lower() == 'e-':
            imgs[ii] /= gain # [e-]
            hdrs[ii][1]['BUNIT'] = 'e-'
        # From Schneider et al. : let's adopt this for all STIS images with
        # GAINCCD = 4 (agreed with Paul).
        # 1 count s-1 pixel-1 (with GAIN=4) = 4.55 x 10-7 Jy
        elif unitEnd.lower() == 'jy':
            # imgs[ii] /= (exptime/nCombine) # WRONG!!! Paul confirmed this is incorrect.
            imgs[ii] /= exptime # convert to counts s^-1 first
            if gain == 4.016:
                imgs[ii] *= 4.55e-7 # [Jy]
            else:
                print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
                sys.exit(1)
        elif unitEnd.lower() == 'jy arcsec-2':
            imgs[ii] /= exptime # convert to counts s^-1 first
            if gain == 4.016:
                imgs[ii] *= (4.55e-7 / pscale**2) # [Jy arcsec-2]
            else:
                print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
                sys.exit(1)
        elif unitEnd.lower() == 'mjy':
            imgs[ii] /= exptime # convert to counts s^-1 first
            if gain == 4.016:
                imgs[ii] *= 4.55e-7 * 1e3 # [mJy]
            else:
                print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
                sys.exit(1)
        elif unitEnd.lower() == 'mjy arcsec-2':
            imgs[ii] /= exptime # convert to counts s^-1 first
            if gain == 4.016:
                imgs[ii] *= (4.55e-7 * 1e3 / pscale**2) # [mJy arcsec-2]
            else:
                print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
                sys.exit(1)

    return imgs


def convert_intensity_stis(img, unitStart=None, unitEnd='counts s-1',
                           gain=None, exptime=None, nCombine=None,
                           photflam=None, stmag=None, pscale=0.0506):
    """
    Convert image from one set of intensity units (aka "flux) to another).
    Possible unit strings are 'counts', 'counts s-1', 'e-', 'jy', 'mjy',
    'jy arcsec-2', or 'mjy arcsec-2'.

    Conversion from counts s-1 to Jy for gain=4.016 adopted from
    Schneider+ 20 () after discussion with P. Kalas.

    Inputs:
        imgs: list of 2-D image arrays to be converted.
        hdrs: list of headers matching imgs in length and order.
        unitStart: str, units of input image. (Default) None will use BUNIT key.
        unitEnd: str, units to output the image in. (Default is 'counts s-1')
        gain: float, detector gain in [counts/e-].
        exptime: float, constituent image exposure time in [s].
        nCombine: int, number of constituent images with exposure time exptime.
        pscale: float, pixel scale of images in [arcsec/pixel]. Defaults
            to STIS' 0.0506 arcsec/pixel.
    """

    imgConv = img.copy()

    # First, convert units to COUNTS as a uniform starting place.
    if unitStart == 'counts':
        pass
    elif unitStart == 'counts s-1':
        # imgConv *= (exptime/nCombine) # <-- THIS WAY IS WRONG!!! Paul confirmed this is incorrect.
        imgConv = imgConv*exptime # [counts]
    elif unitStart == 'e-':
        imgConv = imgConv*gain
    else:
        raise ValueError("unitStart must be 'counts', 'counts s-1', or 'e-'")

    # Now do the final unit conversion.
    if unitEnd.lower() == 'counts s-1':
        # Divide by integration time per image (exptime/nCombine).
        # imgConv /= (exptime/nCombine) # <-- WRONG!!! Paul confirmed this is incorrect.
        imgConv /= exptime # [counts s-1]
    elif unitEnd.lower() == 'e-':
        imgConv /= gain # [e-]
    # From Schneider et al. : let's adopt this for all STIS images with
    # GAINCCD = 4 (agreed with Paul).
    # 1 count s-1 pixel-1 (with GAIN=4) = 4.55 x 10-7 Jy
    elif unitEnd.lower() == 'jy':
        # imgConv /= (exptime/nCombine) # <-- WRONG!!! Paul confirmed this is incorrect.
        imgConv /= exptime # convert to counts s^-1 first
        if gain == 4.016:
            imgConv *= 4.55e-7 # [Jy]
        else:
            print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
            sys.exit(1)
    elif unitEnd.lower() == 'jy arcsec-2':
        imgConv /= exptime # convert to counts s^-1 first
        if gain == 4.016:
            imgConv *= (4.55e-7 / pscale**2) # [Jy arcsec-2]
        else:
            print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
            sys.exit(1)
    elif unitEnd.lower() == 'mjy':
        imgConv /= exptime # convert to counts s^-1 first
        if gain == 4.016:
            imgConv *= 4.55e-7 * 1e3 # [mJy]
        else:
            print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
            sys.exit(1)
    elif unitEnd.lower() == 'mjy arcsec-2':
        imgConv /= exptime # convert to counts s^-1 first
        if gain == 4.016:
            imgConv *= (4.55e-7 * 1e3 / pscale**2) # [mJy arcsec-2]
        else:
            print("*** HELP!! No calibration to Jy provided for gains other than 4 (or 4.016). Edit stis_disk_process.convert_intensity() function.")
            sys.exit(1)
    elif unitEnd.lower() == 'stmag':
        imgConv /= exptime
        # Convert to units of flux in [erg cm^-2 sec^-1 Angstrom^-1]
        # From https://www.stsci.edu/files/live/sites/www/files/home/hst/documentation/_documents/stis/stis_dhb_v4.pdf
        if photflam is not None:
            imgConv *= photflam
        else:
            print("*** HELP!! Must provide a photflam value when coverting to contrast units.")
            sys.exit(1)
        # Then convert from flux in those crazy units to STMAG magnitudes.
        imgConv = -2.5*np.log10(imgConv) - 21.1
    elif unitEnd.lower() == 'contrast':
        imgConv /= exptime
        # Convert to units of flux in [erg cm^-2 sec^-1 Angstrom^-1]
        # From https://www.stsci.edu/files/live/sites/www/files/home/hst/documentation/_documents/stis/stis_dhb_v4.pdf
        if photflam is not None:
            imgConv *= photflam
        else:
            print("*** HELP!! Must provide a photflam value when coverting to contrast units.")
            sys.exit(1)
        # Then convert the reference source STMAG to the same crazy flux units
        fluxRef = utils.stmag_to_flux(stmag)
        # Divide by the image fluxes by the reference source's flux.
        imgConv /= fluxRef

    return imgConv
