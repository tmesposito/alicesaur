#!/usr/bin/env python

import os
import sys
import pdb
import getpass
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.io import ascii, fits
from astropy import table
from astropy import wcs
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Internal imports
import alicesaur
from alicesaur.psfsub import stis_psfsub
from alicesaur import utils
from alicesaur.calibration.bad_pix import fix_bad_pix
from alicesaur.calibration.align import find_star_radon, shift_pix_to_pix
from alicesaur.calibration.flux import convert_intensity
from alicesaur.improcess.mask import mask_exclusions, mask_spikes_offaxis


class Pipeline(object):
    """
    Base reduction pipeline class for Hubble Space Telescope image processing.
    """

    debug = False
    # Locate the path to the alicesaur package. Prioritize the user environment
    # variable ALICESAUR_HOME over retrieving the path from imported modules.
    if os.environ.get('ALICESAUR_HOME') not in [None, '']:
        packageHome = os.path.abspath(os.environ.get('ALICESAUR_HOME'))
    else:
        packageHome = os.path.abspath(alicesaur.__path__[0])

    # Default to STIS instrument.
    instrument = 'stis'
    # Image plate scale.
    pscale = 0.0507 # [arcsec/pixel]
    occultMaskPath = os.path.join(packageHome, 'masks',
                                  'mask_stis_occulters_sx2_bar_wedgeB.fits')

    def __init__(self, **kwargs):

        # Define attributes with kwargs items
        for key, val in kwargs.items():
            setattr(self, key, val)

        if hasattr(self, 'dataDir'):
            # Ensure trailing slash in data directory name.
            self.dataDir = os.path.join(os.path.expanduser(self.dataDir),'')
        else:
            self.dataDir = None

        # Load dataset info and reduction parameters from info.json.
        self.load_info_json(self.dataDir)


    def load_info_json(self, infoDir):
        """
        Load dataset info and reduction parameters from an info.json file.
        """
        try:
            infos = glob(os.path.join(infoDir, "info.json"))
            if len(infos) < 1:
                infos = glob(os.path.join(infoDir, "../info.json"))
            with open(infos[0]) as ff:
                self.info = json.load(ff)
            self.infoPath = os.path.normpath(infos[0])
            print("\nLoaded info from {}".format(self.infoPath))
        except:
            self.info = {}
            self.infoPath = ''
            print(f"\nFailed to load info.json from directory {infoDir}. "
                  + "Filling it with default values.")

        # Set info dict defaults if overriding values were not given in the
        # json file.
        self.info.setdefault('targetName', '')
        self.info.setdefault('psfRefName', '')
        self.info.setdefault('obsLogPath', '')
        self.info.setdefault('diskPA_deg', 0.)
        self.info.setdefault(self.obsMode.lower(), {})
        # Background sampling region centers for science and reference images.
        self.info[self.obsMode].setdefault('bgCen_yx', None)
        # except KeyError as ee:
        #     print("Error reading info.json at {}. Check for typos and missing keywords. Exiting.\n".format(ee))
        #     sys.exit(1)
        self.info[self.obsMode].setdefault('bgCenRef_yx', None)
        # if not (np.all(bgCenRef is None) | (bgCenRef == '')):
        #     bgCenRef = np.array(bgCenRef.split(' '), dtype=float)
        # else:
        #     bgCenRef = bgCen
        self.info[self.obsMode].setdefault('bgCenFinal_yx', None)
        # Background sampling radius [pix]
        self.info[self.obsMode].setdefault('bgRadius', 40)
        # Diffraction spike mask width [pix]
        self.info[self.obsMode].setdefault('spWidth', 5)
        self.info[self.obsMode].setdefault('radProfSub', {})
        # Radon transform inner working angle [pix]
        if self.obsMode.lower() in ['wedgeb1.8']:
            self.info[self.obsMode].setdefault('radonIWA', 70)
        else:
            self.info[self.obsMode].setdefault('radonIWA', 30)
        # starToUse is the optional forced star position
        self.starToUse = self.info[self.obsMode].setdefault('starForce_yx',
                                                            None)
        if self.starToUse is not None:
            self.forceStar = True
            self.starToUse = np.array(self.starToUse)
        else:
            self.forceStar = False
         # Masked exclusion region definitions for Science and Reference images
        self.exclusions = self.info[self.obsMode].setdefault('exclude', {})
        self.exclusionsSci = self.exclusions.setdefault('sci', {})
        self.exclusionsRef = self.exclusions.setdefault('ref', {})
        # PSF subtraction inner and outer radii.
        self.exclusionsSci.setdefault('r_in', 5)
        self.exclusionsSci.setdefault('r_out', 70)

        # Set the background attributes.
        # Background sampling center and radius for science images.
        bgCen = self.info[self.obsMode]['bgCen_yx']
        if not (np.all(bgCen is None) | (bgCen == '')):
            self.bgCen = np.array(bgCen.split(' '), dtype=float)
        else:
            self.bgCen = None
        self.bgRadius = self.info[self.obsMode]['bgRadius'] # [pix]
        # Background sampling center for reference images.
        bgCenRef = self.info[self.obsMode]['bgCenRef_yx']
        if not (np.all(bgCenRef is None) | (bgCenRef == '')):
            self.bgCenRef = np.array(bgCenRef.split(' '), dtype=float)
        else:
            self.bgCenRef = None

        return


    def load_imgs(self, suffix='sx2'):
        """
        List of arrays containing data [0] and headers [1].

        suffix: str suffix of FITS files to summarize.
        """


        if not os.path.exists(self.dataDir):
            print("\n!! **Directory {} does not exist ** !!\n".format(self.dataDir))
            raise OSError

        fl = np.sort(glob(self.dataDir + '*_{}.fits*'.format(suffix)))
        imgsHdrs = [[],[]]
        targs = []

        if len(fl) == 0:
            print("HELP!!! No FITS files found at {}".format(self.dataDir + '*_{}.fits*'.format(suffix)))

        for ii, ff in enumerate(fl):
            with fits.open(ff) as hdu:
                imgs = [hdu[jj].data for jj in range(len(hdu))]
                hdrs = [hdu[jj].header for jj in range(len(hdu))]
                imgsHdrs[0].append(imgs)
                imgsHdrs[1].append(hdrs)
                targs.append(hdrs[0]['TARGNAME'].lower())

        return imgsHdrs, np.array(targs), fl
    
    
    def summarize_obs(self, suffix='sx2', dsName=None):
        """
        Summarize FITS header info in a table.

        suffix: str suffix of FITS files to summarize.
        dsName: str optional name of dataset for log filename; defaults to the
            lowest level directory of dataDir.
        """

        if not os.path.exists(self.dataDir):
            print("\n!! **Directory {} does not exist ** !!\n".format(self.dataDir))
            raise OSError
    
        fl = np.sort(glob(self.dataDir + '*_{}.fits*'.format(suffix)))
    
        rows = []
        col_names = ['I', 'FILENAME', 'TARGNAME', 'ORIENTAT', 'TDATEOBS', 'TTIMEOBS', 'PROPAPER', 'NCOMBINE',
                     'EXPTIME', 'NDATAARR']
        col_dtypes = ['int32', 'U80', 'U40', 'U40', 'U40', 'U40', 'U40', 'int32',
                      'float', 'int32']
    
        for ii, ff in enumerate(fl):
            hdr0 = fits.getheader(ff, ext=0)
            hdr1 = fits.getheader(ff, ext=1)
            # Get number of data arrays in _flt.fits file.
            try:
                ff_flt = ff.split(suffix)[0] + 'flt.fits'
                hdr_flt = fits.getheader(ff_flt, ext=0)
                nDataDim = hdr_flt['NEXTEND'] + 1
            except:
                nDataDim = -1
            rows.append((ii+1, os.path.split(ff)[-1], hdr0.get('TARGNAME'), hdr1.get('ORIENTAT'), hdr0.get('TDATEOBS'),
                         hdr0.get('TTIMEOBS'), hdr0.get('PROPAPER'),
                         hdr1.get('NCOMBINE'), hdr1.get('EXPTIME'), nDataDim))
    
        sum_table = table.Table(rows=rows, names=col_names, dtype=col_dtypes)
    
        if dsName is None:
            dsName = os.path.split(self.dataDir[:-1])[-1]
        table_name = 'obs_log_{}.csv'.format(dsName)
        sum_table.write(self.dataDir + table_name, format='csv', overwrite=True)
        print("Wrote observation summary table to {}".format(self.dataDir + table_name))
    
        return
    
    
    def load_obs_log(self, logPath=None):
        """
        """

        if logPath is not None:
            logList = [logPath]
        else:
            logList = glob(self.dataDir + 'obs_log*.csv')
            if len(logList) == 0:
                logList = glob(self.dataDir + '../obs_log*.csv')

        try:
            self.obsTable = ascii.read(os.path.expanduser(logList[0]),
                                       delimiter=",",
                                       header_start=0,
                                       data_start=1,
                                       guess=True)
        except:
            self.obsTable = None

        return self.obsTable


    def combineOrbitImgs(self, imgs, orientats):
        """
        Combine (nanmean) all images from a single orbit for a given target.
        """
        # Maintain the input order of the sci and ref images.
        orientatsUnique, indsUnique = np.unique(orientats, return_index=True)
        sameOrder = np.argsort(indsUnique)
        orientatsUnique = orientatsUnique[sameOrder]
    
        orientatsSci = np.unique(orientats[self.sciInds])
        orientatsRef = np.unique(orientats[self.refInds])

        sciIndsNew = np.array([np.where(orientatsUnique == oo)[0][0] for oo in orientatsSci])
        refIndsNew = np.array([np.where(orientatsUnique == oo)[0][0] for oo in orientatsRef])
        self.sciInds = sciIndsNew
        self.refInds = refIndsNew

        combineImgs = np.zeros((len(orientatsSci) + len(orientatsRef), imgs.shape[1], imgs.shape[2]))

        for ii, orient in enumerate(orientatsSci):
            indsOrbit = np.where(orientats == orient)[0]
            combineImgs[np.where(orientatsUnique == orient)[0]] = np.nanmean(imgs[indsOrbit], axis=0)

        for ii, orient in enumerate(orientatsRef):
            indsOrbit = np.where(orientats == orient)[0]
            combineImgs[np.where(orientatsUnique == orient)[0]] = np.nanmean(imgs[indsOrbit], axis=0)

        return combineImgs, orientatsUnique


    def find_star(self, imgs, mask):
        """
        Locate the target star position in pixel coordinates. The default is to
        use a Radon transform-based algorithm to locate the center of the
        occulted primary star via its diffraction spikes.
        """
        
        # Find star.
        stars = []

        for ii, im in enumerate(imgs):
            # Mask out the occulters before doing the radon transform.
            imgIter = im + mask
            imgIter[imgIter < -1e3] = 0.
            # Do a radon transform to find star from diffraction spike pattern.
            # radonIWA = 30 pix generally good for bar10
            # sp_width = 30 pix generally good for bar10
            if self.noRadon or self.forceStar:
                stars.append(self.starFromWCS)
                print("Assuming all stars at {}".format(self.starFromWCS))
            else:
                print("\nRadon transforming image {}...".format(ii))
# FIX ME!!! This 106906 block is a kludge specific to Esposito et al. in prep
# that needs to be removed eventually.
                # HD 106906 wedgeb1.8 is special case with dithering along
                # wedge, so handle that here.
                if (self.targ == 'HD-106906') and (self.obsMode == 'wedgeb1.8'):
                    stars.append(find_star_radon(imgIter,
                                        self.starFromWCS_list[ii],
                                        self.spikeAngles, IWA=self.radonIWA,
                                        sp_width=20, r_mask=None)) # [pixels] y,x
                else:
                    stars.append(find_star_radon(imgIter,
                                        self.starFromWCS,
                                        self.spikeAngles, IWA=self.radonIWA,
                                        sp_width=20, r_mask=None)) # [pixels] y,x

        self.stars = np.array(stars) # [pixels] y,x

        return


# FIX ME!!! noOffsetDatasets is a kludge specific to Esposito et al. in prep
# that needs to be removed eventually.
    def subtract_background(self, noOffsetDatasets=[]):
        """
        Subtract a global background from each image in self.workingImgs.
        """
        bgs = []
        for ii, im in enumerate(self.workingImgs):
            # Work on a source-masked copy of the aligned image.
            imBgTmp = im.copy()
            # imBgTmp[sourceMasks[ii]] = np.nan
            # Assume all reference images are taken at the same PA.
            if ii not in self.refInds:
                bgCenRot = utils.rotate_yx(self.bgCen, self.alignStar,
                                           self.orientats[ii])
# FIX ME!!! For modern system of padding images, don't add the star offsets to the bg positions.
                if self.targ not in noOffsetDatasets:
                    bgCenRot += self.alignStarOffsets[ii]
            # Still need to offset the reference image for these modern cases
            # because bg positions are chosen from the raw images.
            else:
                bgCenRot = self.bgCenRef + self.alignStarOffsets[ii]
            imSub, bg = utils.subtract_bg(imBgTmp, bgCenRot, self.bgRadius)
            # Can't use subtract_bg image output here because it is masked.
            self.workingImgs[ii] = im - bg
            bgs.append(bg)

        self.bgs = np.array(bgs)

        return


    def derotate(self, imgs, orientats, cens):
    
        rotImgs = []
        for ii in range(len(imgs)):
            angle = orientats[ii]
            rotImgs.append(utils.rotate_array(imgs[ii], cens[ii], angle,
                                              preserve_nan=True, cval=1))
    
        return np.array(rotImgs)
    
    
    def run(self):
        """
        Run the complete reduction pipeline from start to finish.
        """

# TEMP!!! Clean this up later to just use the self attributes directly.
        dataDir = self.dataDir
        obsMode = self.obsMode
        targ = self.targ
        ann = self.ann
        saveFinal = self.saveFinal
# FIX ME!!! Convert suff to input arg.
        suff = 'sx2'
        psfSubMode = self.psfSubMode
        noCombine = self.noCombine
        pad = not self.noPad

        roiY = [-200, 200]
        roiX = [-200, 200]

# FIX ME!!! Instrument-specific stuff here needs to go into instrument modules.
        # Angles at which diffraction spikes occur in STIS data [deg].
        self.spikeAngles = np.array([44.9, 134.7]) # [deg] clockwise from 0 at +X


        # # Summarize the dataset info from FITS headers.
        # if len(glob(dataDir + 'obs_log*.csv')) < 1:
        #   summarize_obs(dataDir, suffix=suff, dsName=None)

        # Fetch dataset info and reduction parameters from info.json.
        info, infoPath = self.info, self.infoPath

        # Set reduction parameters based on info.json if overriding values were not given
        # as input args.
        if targ in ['', None]:
            targ = self.info['targetName']
        psfRefName = self.info['psfRefName']
        if self.spWidth is None:
            self.spWidth = self.info[self.obsMode]['spWidth'] # diffraction spike mask width [pix]
        self.radonIWA = self.info[self.obsMode]['radonIWA']
        radonIWA = self.radonIWA
        exclusions = self.info[self.obsMode]['exclude'] # masked region definitions
        exclusionsSci = self.exclusionsSci # science masking
        sub_r_in = self.exclusionsSci['r_in'] # PSF subtraction inner radius
        sub_r_out = self.exclusionsSci['r_out'] # PSF subtraction outer radius

        # Load observation log.
        self.load_obs_log(logPath=info.get('obsLogPath'))

        # Load images to be processed.
        # Returns list where [0] = data from each fits, [1] = headers from each fits.
        # Each item in [0] and [1] is also a list of arrays.
        print("Loading data...")
        imgsHdrs, targs, fl = self.load_imgs(suffix=suff)

        # Get input image intensity units.
        bunit = imgsHdrs[1][0][1]['BUNIT']

        # Separate science frames from PSF reference frames via the target name.
        self.sciInds = np.where(np.char.lower(targs) == targ.lower())[0]
        self.refInds = np.where(np.char.lower(targs) != targ.lower())[0]
        # Number of input science and reference frames (before any combinations)
        nSci = np.size(self.sciInds)
        self.nRef = np.size(self.refInds)

        print("\nScience image indices: {}".format(self.sciInds))
        print("Ref image indices ({}): {}\n".format(targs[self.refInds], self.refInds))

        # Get UT observation date, proposed aperture name, orientat angles, and
        # rough star coordinates from headers.
        obsDate = imgsHdrs[1][self.sciInds[0]][0]["TDATEOBS"] # [UT]
        propAper = imgsHdrs[1][self.sciInds[0]][0]["PROPAPER"]
        orientats = np.array([imgsHdrs[1][ii][1]["ORIENTAT"] for ii in range(len(imgsHdrs[0]))])
        self.orientats = orientats

        exptimes_s = [imgsHdrs[1][ii][0].get('TEXPTIME', -1) for ii in self.sciInds] # [s]
        photflam_avg = np.mean([imgsHdrs[1][ii][0].get('PHOTFLAM', -1.) for ii in self.sciInds])

        # Either force star position or estimate based on headers.
        if self.forceStar:
            self.starFromWCS = self.starToUse
        else:
            self.starFromWCS_list = []
            for ii, hdr in enumerate(imgsHdrs[1]):
                # Get estimate of star position from target RA/Dec and WCS in header.
                ww = wcs.WCS(hdr[1])
                targRA = hdr[0]['RA_TARG']
                targDec = hdr[0]['DEC_TARG']
                self.starFromWCS_list.append(ww.wcs_world2pix([[targRA, targDec]], 0)[0][::-1]) # [pixels] y,x
            self.starFromWCS = np.mean(self.starFromWCS_list, axis=0)

        # Check number of images per fits file.
        nImgsPerFits = [len(imgsHdrs[0][ii]) for ii in range(len(imgsHdrs[0]))]
        print("Data arrays per fits file: {}".format(nImgsPerFits))

        # Separate images into their own 3-d array. Includes Sci and Ref images.
        sciImgs = np.array([imgsHdrs[0][ii][1] for ii in range(len(imgsHdrs[0]))])


    # ========== FIX BAD PIXELS ========== #
        # Fix bad pixels iteratively in all images.
        if not self.noFixPix:
            intenseROI = 460
            sciImgs = fix_bad_pix(sciImgs,
                                  intensify=[int(self.starFromWCS[0])-intenseROI//2,
                                             int(self.starFromWCS[1])-intenseROI//2,
                                             intenseROI, intenseROI])

    # ========== CALIBRATE FLUX ========== #
        # Convert intensity to counts per second.
        newUnit = 'COUNTS S-1'
        # newUnit = 'mJy arcsec-2'
        sciImgs = convert_intensity(sciImgs, imgsHdrs[1], unitEnd=newUnit,
                                    pscale=self.pscale) # [counts/s]
        bunit = newUnit
        print("Converted image intensity units to {}".format(newUnit))

    # ========== AVERAGE IMAGES BY ORBIT ========== #
        # Combine individual exposures in an orbit to make one image.
        # Redefine orientats and indices based on these combined images.
        if (not noCombine) and (psfSubMode.lower() not in ['pyklip-rdi']):
            orbitImgs, orientats = self.combineOrbitImgs(sciImgs, orientats)
        else:
            orbitImgs = sciImgs.copy()

        # Apply distortion correction if needed (sx2 already corrected/"rectified")
        if suff not in ['sx2']:
    # FIX ME!!! Add distortion correction step
            pass

        # Mask out occulting wedges (later replacing with NaN values).
        if self.instrument == 'stis':
            occultMask = fits.getdata(self.occultMaskPath)
        else:
            print("***HELP!!! Occulter masks are not implemented yet for "\
                  "instruments other than STIS")
            occultMask = np.zeros(orbitImgs[0].shape)
        occultMask[occultMask < 0] = -1.e4


    # ==== ALIGN IMAGES TO COMMON STAR POSITION ==== #

        # Find the occulted star's coordinates in every image.
        self.find_star(orbitImgs, occultMask)

        # Define new aligned star position.
        if pad:
            matSize = np.array([2048, 2048])
            alignStar = matSize//2
        else:
            matSize = None
            alignStar = np.round(self.stars[0])
        self.alignStar = alignStar

        # Register images to align stars.
        # Also pad the images to matSize if self.noPad is not set.
        print("\nAligning images and masks to common star position...")
        alignImgs = []
        alignMasks = []
        for ii in tqdm(range(len(orbitImgs)), desc="Aligned images"):
            alignImgs.append(shift_pix_to_pix(orbitImgs[ii], self.stars[ii],
                                              finalYX=self.alignStar, outputSize=matSize,
                                              order=3, fill=0.))
            alignMasks.append(shift_pix_to_pix(occultMask, self.stars[ii],
                                               finalYX=self.alignStar, outputSize=matSize,
                                               order=1, fill=-1e4))
        alignImgs = np.array(alignImgs)
        alignMasks = np.array(alignMasks)
        # Preserve original star positions before overwriting stars with new
        # aligned position. Keep the offsets around for posterity.
        self.starsOriginal = self.stars.copy()
        self.stars[:] = self.alignStar
        self.alignStarOffsets = self.stars - self.starsOriginal

        if self.debug:
            fig = plt.figure(3)
            for ii in range(len(alignImgs)):
                st = np.round(self.stars[ii]).astype(int)
                fig.clf()
                ax = plt.subplot(111)
                ax.imshow(alignImgs[ii][st[0]+roiY[0]:st[0]+roiY[1],
                                        st[1]+roiX[0]:st[1]+roiX[1]],
                          norm=SymLogNorm(linthresh=1., linscale=1.,
                                          vmin=0., vmax=5000.),
                          extent=[st[1]+roiX[0], st[1]+roiX[1],
                                  st[0]+roiY[0], st[0]+roiY[1]])
                ax.scatter(x=st[1], y=st[0], marker='+', s=60, c='m')
                ax.set_title("Aligned Image {}".format(ii))
                plt.draw()
                pdb.set_trace()

        noOffsetDatasets = ['V-AK-SCO', 'GSC-07396-00759', 'HD-106906', 'HD-111161',
                            'HD-114082', 'HD-115600', 'HD-117214',
                            'HD-129590', 'HD-145560', 'HD-146897']


    # ==== BACKGROUND SUBTRACTION ==== #

        # Subtract background/sky.
# FIX ME!!! Assign workingImgs higher up eventually.
        self.workingImgs = alignImgs
        if self.bgCen is not None:
            print("\nSubtracting background from all images...")
            self.subtract_background(noOffsetDatasets=noOffsetDatasets)
            print("Background means subtracted: {}".format(self.bgs))
        else:
            print("Skipping background subtraction (bgCen is None)")

    # # BG TESTING!!!
    #     plt.figure(5)
    #     plt.clf()
    #     for ii in self.workingImgs:
    #         plt.imshow(ii, norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=0, vmax=50))
    #         plt.ylim(1024-100, 1024+101)
    #         plt.xlim(1024-200, 1024+201)
    #         plt.draw()
    #         pdb.set_trace()


    # ==== CREATE IMAGE MASKS ==== #

        psfsubOnSpikesOnly = False

        print("\nMaking PSF subtraction masks...")
        # Make mask for occulters and diffraction spikes.
        radii = utils.make_radii(self.workingImgs[0], self.alignStar)
        masks = []
        for ii, img in enumerate(self.workingImgs):
            spikemask = utils.make_spikemask_stis(self.workingImgs[0], self.alignStar,
                                                  self.spikeAngles, width=self.spWidth)
    # FIX ME!!! Combine spikemask with occulter mask here.
            masks.append(spikemask)
        masks = np.array(masks)
        psfSubMasks = masks.copy()
        # Make a different set of masks that does not do the radial masking.
        if psfsubOnSpikesOnly:
            sourceMasks = np.zeros(masks.shape).astype(bool)
        else:
            sourceMasks = masks.copy()
        # Make special masks for background star charge bleed.
        bgStarMasks = []
        # Specialize science masks.
        for ii, ind in tqdm(enumerate(self.sciInds), desc="Science masks"):
    # FIX ME!!! May want to move this radial masking to the RDI PSF subtraction function
    # so it doesn't conflict with pyKLIP?
            # Mask out the occulted sections too, established earlier as very negative valued.
            sourceMasks[ind][alignMasks[ind] < 0] = np.nan
            # Now mask all of the excluded sources given in "exclude" json key.
            if targ not in noOffsetDatasets:
                sourceMasks[ind] = mask_exclusions(mask=sourceMasks[ind],
                                       exclusions=exclusionsSci,
                                       cen=self.alignStar, cenOffset=self.alignStarOffsets[ind],
                                       paOffset=-1*orientats[ind], spikeAngles=self.spikeAngles)
            else:
                sourceMasks[ind] = mask_exclusions(mask=sourceMasks[ind],
                                       exclusions=exclusionsSci,
                                       cen=self.alignStar, cenOffset=np.zeros(2),
                                       paOffset=-1*orientats[ind], spikeAngles=self.spikeAngles)
            # Now make a new mask by folding in the radial masking, specifically for
            # PSF subtraction only.
    # TEMP!!!
            # psfSubMasks[ind][radii >= sub_r_out] = np.nan
            # psfSubMasks[ind][radii < sub_r_in] = np.nan
            # psfSubMasks[ind] += sourceMasks[ind]

    # # TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
            if psfsubOnSpikesOnly:
                # psfSubMasks[ind] = ~masks[ind]
                try:
                    spikePSF_rmin, spikePSF_rmax = info[obsMode]['spikePSF_rminmax']
                    psfSubMasks[ind][radii >= spikePSF_rmax] = np.nan
                    psfSubMasks[ind][radii < spikePSF_rmin] = np.nan
                except:
                    psfSubMasks[ind][radii >= sub_r_out] = np.nan
                    psfSubMasks[ind][radii < sub_r_in] = np.nan
                psfSubMasks[ind] += sourceMasks[ind].copy()
    # # END TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
                sourceMasks[ind] += masks[ind].copy()
            else:
                psfSubMasks[ind][radii >= sub_r_out] = np.nan
                psfSubMasks[ind][radii < sub_r_in] = np.nan
                psfSubMasks[ind] += sourceMasks[ind].copy()
    
    # TEMP!!! Force in a bg star mask for HD 111161 wedge
            if (targ == 'HD-111161') & (obsMode == 'wedgeb1.0'):
                bgStarCen = np.array([987, 763]) - self.alignStar
                bgStarCenRot = np.array([np.cos(np.radians(orientats[ind]))*bgStarCen[0] - np.sin(np.radians(orientats[ind]))*bgStarCen[1],
                                         np.sin(np.radians(orientats[ind]))*bgStarCen[0] + np.cos(np.radians(orientats[ind]))*bgStarCen[1]], dtype=int)
                bgStarCenRot += self.alignStar
                bgStarMask = np.zeros(self.workingImgs[0].shape).astype(bool)
                bgStarMask[:, bgStarCenRot[1]-15:bgStarCenRot[1]+15] = True
                bgStarMasks.append(bgStarMask)
                
                psfSubMasks[ind] += bgStarMask
                sourceMasks[ind] += bgStarMask
            
            # Add off-axis diffraction spike masks to the other masks.
            # IMPORTANT: cen here is in the padded image coordinate frame-- NOT the original.
            for excl in exclusionsSci.setdefault('spikes_yxr', []):
                maskSpikesOffaxis = mask_spikes_offaxis(np.zeros(alignMasks[ind].shape), excl,
                                                    cen=self.alignStar,
                                                    cenOffset=None,
                                                    paOffset=-1*orientats[ind],
                                                    spikeAngles=self.spikeAngles)
                maskSpikesOffaxis *= -1e4
                alignMasks[ind] += maskSpikesOffaxis
                psfSubMasks[ind][maskSpikesOffaxis < 0] = True
                sourceMasks[ind][maskSpikesOffaxis < 0] = True

            if self.debug:
                plt.figure(4)
                plt.clf()
                plt.imshow(alignMasks[ind])
                plt.title("Occulter ('align') mask")
                plt.draw()

                pdb.set_trace()

        # Specialize reference masks. Default radius masks to match science masks.
        exclusionsRef = exclusions.setdefault('ref', {})
        # if exclusionsRef.get('r_out') is None:
        #   exclusionsRef['r_out'] = exclusionsSci.get('r_out')
        # if exclusionsRef.get('r_in') is None:
        #   exclusionsRef['r_in'] = exclusionsSci.get('r_in')
        for ii, ind in tqdm(enumerate(self.refInds), desc="Reference masks"):
    # TEMP!!! TEST ONLY PSF SUBTRACTING BASED ON DIFFRACTION SPIKES.
            if psfsubOnSpikesOnly:
                psfSubMasks[ind] = np.zeros(psfSubMasks[ind].shape).astype(bool)
            # psfSubMasks[ind][radii >= exclusionsRef.setdefault('r_out', 70)] = np.nan
            # psfSubMasks[ind][radii < exclusionsRef.setdefault('r_in', 35)] = np.nan
            psfSubMasks[ind][alignMasks[ind] < 0] = np.nan
            # Don't offset PA of reference mask because coords should already be
            # given in rotated frame.
            # Always offset the coordinates based on the new aligned star center
            # for reference images because those mask coordinates are only ever
            # measured from the raw images.
            # Special case of HD 106906 wedgeb1.8 has two dither positions on same
            # wedge, so handle that offset correctly.
            if (targ == 'HD-106906') and (obsMode == 'wedgeb1.8'):
                psfSubMasks[ind] = mask_exclusions(mask=psfSubMasks[ind],
                                            exclusions=exclusionsRef,
                                            cen=self.alignStar, cenOffset=self.alignStarOffsets[self.refInds[0]],
                                            paOffset=0, spikeAngles=self.spikeAngles)
            else:
                psfSubMasks[ind] = mask_exclusions(mask=psfSubMasks[ind],
                                            exclusions=exclusionsRef,
                                            cen=self.alignStar, cenOffset=self.alignStarOffsets[ind],
                                            paOffset=0, spikeAngles=self.spikeAngles) #-1*orientats[ind])
    
            if self.debug:
                for ii in range(len(psfSubMasks)):
                    plt.figure(5)
                    plt.clf()
                    plt.imshow(psfSubMasks[ii])
                    plt.title(f"PSF Subtraction mask: img {ii}")

                    plt.figure(6)
                    plt.clf()
                    plt.imshow(self.workingImgs[ii],
                                norm=SymLogNorm(linthresh=0.01, linscale=1,
                                                vmin=0, vmax=100))
                    plt.title(f"Aligned image: img {ii}")

                    plt.figure(7)
                    plt.clf()
                    plt.imshow(self.workingImgs[ii]*~sourceMasks[ii],
                                norm=SymLogNorm(linthresh=0.01, linscale=1,
                                                vmin=0, vmax=100))
                    plt.title(f"Aligned image with PSF Subtraction mask: img {ii}")

                    pdb.set_trace()


    # ======== PSF SUBTRACTION ======== #

        # Basic RDI PSF subtraction.
        if psfSubMode.lower() == 'rdi':
            print("Performing RDI PSF subtraction...")
            rmin = 1 # PSFsub masking supercedes this value.
            getRadProf = info[obsMode].get('radProfSub')
            if getRadProf and (getRadProf is not None):
                radProfPaList = info[obsMode]['radProfSub'].setdefault('paList', [0])
                if radProfPaList is not None:
                    radProfPaList = np.array(radProfPaList)
                    subRadProf = True
                else:
                    subRadProf = False
                radProfPaHW = info[obsMode]['radProfSub'].setdefault('paHW', 50)
                radProfMax = info[obsMode]['radProfSub'].setdefault('rMax', 200)
            else:
                radProfPaList = None
                radProfPaHW = None
                radProfMax = None
                subRadProf = False
            alignMasked = []
            psfSubImgs, refScaleFactors = stis_psfsub.rdi_subtract_psf(
                                            self.workingImgs[self.sciInds],
                                            self.workingImgs[self.refInds],
                                            psfSubMasks[self.sciInds],
                                            psfSubMasks[self.refInds],
                                            self.stars[self.sciInds], C0=-1.,
                                            rmin=rmin, rmax=sub_r_out,
                                            ann=ann, orientats=orientats[self.sciInds],
                                            radProfPaList=radProfPaList,
                                            radProfPaHW=radProfPaHW,
                                            radProfMax=radProfMax,
                                            radProfMasks=sourceMasks[self.sciInds],
                                            subRadProf=subRadProf,
                                            bgCen=self.bgCen, bgRadius=self.bgRadius)
            print(f"Ref scale factors by science image: {refScaleFactors}")
        # Basic ADI PSF subtraction.
        elif psfSubMode.lower() == 'adi':
            print("Performing ADI PSF subtraction...")
            psfSubImgs = stis_psfsub.adi_subtract_psf()
        # PyKLIP RDI subtraction.
        elif psfSubMode.lower() == 'pyklip-rdi':
            print("Performing pyKLIP RDI PSF subtraction...")
            IWA = 3.
            OWA = 200.
            # Mask out the occulters in the input images.
            alignImgsMasked = self.workingImgs + alignMasks
            alignImgsMasked[alignMasks < 0] = np.nan

            stis_psfsub.do_klip_stis(targ, fl[self.sciInds],
                inputImgs=alignImgsMasked[self.sciInds], inputHdrs=None,
                psfPaths=fl, psfImgs=alignImgsMasked, mode='RDI',
                ann=int(OWA-IWA), subs=1, minrot=0, mvmt=0, IWA=IWA, OWA=OWA,
                numbasis=[1,2,3,5,15], maxnumbasis=None, star=self.stars,
                highpass=False, pre_sm=None, spWidth=8., ps_spWidth=0., PAadj=0.,
                parangs=None, aligned_center=self.alignStar[::-1],
                collapse="mean", prfx=targ,
                save_psf_cubes=False, save_aligned=False, restored_aligned=None,
                lite=False, do_snmap=False, numthreads=4, sufx='', output=False,
                compute_correlation=True)
        # PyKLIP ADI subtraction.
        elif psfSubMode.lower() == 'pyklip-adi':
            print("Performing pyKLIP ADI PSF subtraction...")
            stis_psfsub.do_klip_stis(targ, fl[self.sciInds],
                inputImgs=self.workingImgs[self.sciInds], inputHdrs=None,
                psfPaths=None, psfImgs=None, mode='ADI', 
                ann=1, subs=1, minrot=0, mvmt=5, IWA=10., OWA=200.,
                numbasis=[1,2,3,5,10,15], maxnumbasis=None, star=self.stars[self.sciInds],
                highpass=False, pre_sm=None, spWidth=8., ps_spWidth=0., PAadj=0.,
                parangs=orientats[self.sciInds], aligned_center=self.alignStar[::-1],
                collapse="mean", prfx=targ,
                save_psf_cubes=False, save_aligned=False, restored_aligned=None,
                lite=True, do_snmap=False, numthreads=4, sufx='', output=False,
                compute_correlation=False)
    
        # Final background subtraction.
    # SKIP -- for now.
    
        if psfSubMode.lower() in ["adi", "rdi"]:
            # Apply mask for occulters and diffraction spikes to PSF-subtracted images.
            for ii in range(len(psfSubImgs)):
                aM = alignMasks[self.sciInds][ii].copy()
                if len(bgStarMasks) > 0:
                    aM[bgStarMasks[ii]] = -1
                aM[aM >= 0] = 0
                aM[aM < 0] = 1
                aM = aM.astype(bool)
                psfSubImgs[ii][masks[self.sciInds][ii] + aM] = np.nan
        
            # Derotate and combined PSF-subtracted images.
            print("Derotating PSF-subtracted images...")
            rotImgs = self.derotate(psfSubImgs, orientats[self.sciInds],
                                    self.stars[self.sciInds])
            
            # Also make a copy with images derotated in the wrong direction.
            # Make the widest deltaPA = 90 deg from being fully aligned.
            spreadPAs = orientats[self.sciInds].copy()
            spreadPAs[1:] -= np.arange(1, len(self.sciInds))*(90/(len(self.sciInds)-1))
            rotImgsBkwd = self.derotate(psfSubImgs, spreadPAs,
                                        self.stars[self.sciInds])
            
            
            # Optimize combination based on background levels, if needed.
            print("Combining derotated images...")
            finalImg = np.nanmean(rotImgs, axis=0)
            bkwdImg = np.nanmedian(rotImgsBkwd, axis=0)
            
            # plt.figure(1)
            # plt.clf()
            # for ii in rotImgs:
            #     plt.imshow(ii, norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=0, vmax=50))
            #     plt.ylim(1024-100, 1024+101)
            #     plt.xlim(1024-200, 1024+201)
            #     plt.draw()
            #     pdb.set_trace()
            # 
            # pdb.set_trace()
            
            # Subtract radial profile from final combined image to remove
            # residual halo.
            subFinalRadProf = False
            if (info[obsMode].get('radProfSub') is not None):
                if info[obsMode]['radProfSub'].get('postCombine') == 'True':
                    tmpImg = finalImg.copy()
                    # tmpImg = bkwdImg.copy()
                    # Remove disk mask from exclusions.
                    exclusionsFinal = exclusionsSci.copy()
                    exclusionsFinal['rect_cenYX_widthYX_angleDeg'] = []
                    finalMask = np.zeros(finalImg.shape, dtype=bool)
                    if targ not in noOffsetDatasets:
                        finalMask = mask_exclusions(mask=finalMask,
                                       exclusions=exclusionsFinal,
                                       cen=self.alignStar, cenOffset=self.alignStar - self.starsOriginal[0],
                                       paOffset=0, spikeAngles=self.spikeAngles)
                    else:
                        finalMask = mask_exclusions(mask=finalMask,
                                       exclusions=exclusionsFinal,
                                       cen=self.alignStar, cenOffset=np.zeros(2),
                                       paOffset=0, spikeAngles=self.spikeAngles)
                    tmpImg[finalMask.astype(bool)] = np.nan
                    meanRadProf = stis_psfsub.measure_mean_radial_prof(tmpImg, self.alignStar,
                                                           paList=info[obsMode]['radProfSub']['paList'],
                                                           paHW=info[obsMode]['radProfSub']['paHW'],
                                                           rMax=info[obsMode]['radProfSub']['rMax'])
                                                           # paList=np.arange(170., 271., 5.) - orientats[ii])
                                                           # paList=np.append(np.arange(20., 31., 5.), np.arange(170., 271., 5.)) - orientats[ii])
                    meanRadProf = np.nan_to_num(meanRadProf, 0)
                    finalImg -= meanRadProf
                    subFinalRadProf = True

                    if self.debug:
                        plt.figure(15)
                        plt.clf()
                        plt.imshow(finalMask)
                        plt.title("Mask for post-combine\nradial profile measurement")

                        plt.figure(16)
                        plt.clf()
                        plt.imshow(tmpImg,
                                   norm=SymLogNorm(linthresh=0.01, linscale=1,
                                                   vmin=-0.1, vmax=10))
                        plt.title("Post-combine image masked\nfor radial profile measurement")

                        plt.figure(17)
                        plt.clf()
                        plt.imshow(meanRadProf,
                                   norm=SymLogNorm(linthresh=0.01, linscale=1,
                                                   vmin=-0.1, vmax=10))
                        plt.title("Post-combine radial profile measured")

                        pdb.set_trace()
            
            # Do one final background subtraction.
            if info[obsMode].get('bgCenFinal_yx') is not None:
                finalImg, bgFinal = utils.subtract_bg(finalImg, np.array(info[obsMode].get('bgCenFinal_yx').split(' '), dtype=float), self.bgRadius)
    
            if saveFinal:
                if newUnit in ['Jy', 'Jy arcsec-2', 'mJy', 'mJy arcsec-2']:
                    saveName = "{}_{}_stis_{}_{}_a{}_final_{}.fits".format(targ, obsDate,
                                                    propAper, psfSubMode.lower(), ann,
                                                    newUnit.replace(' ', '_'))
                else:
                    saveName = "{}_{}_stis_{}_{}_a{}_final.fits".format(targ, obsDate,
                                                    propAper, psfSubMode.lower(), ann)
                # if os.path.isfile(dataDir + saveName): raise IOError
                hdu = fits.PrimaryHDU(data=finalImg.astype('float32'))
                # hdu.header = data_hdr.copy()
                if not 'HISTORY' in hdu.header:
                    try:                
                        hdu.header.add_history('Created by {}'.format(getpass.getuser()))
                    except:
                        hdu.header.add_history('Created by unknown user')
                hdu.header['TARGNAME'] = (targ)
                hdu.header['PSFNAME'] = (psfRefName, 'Reference PSF name')
                hdu.header['FILETYPE'] = ('Final combined image')
                hdu.header['NCOMBINE'] = (nSci, 'Number of images combined')
                hdu.header['INPUTTYP'] = (suff, 'Type of input data')
                hdu.header['PSFSUBMD'] = (psfSubMode, 'PSF-subtraction mode')
                hdu.header['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
                hdu.header['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
                hdu.header['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
                hdu.header['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded mean star center Y')
                hdu.header['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded mean star center X')
                hdu.header['TEXPTIME'] = (np.sum(exptimes_s), 'Total combined integration time in s')
                hdu.header['BUNIT'] = (bunit, 'brightness units')
                hdu.header['PHOTFLAM'] = (photflam_avg, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
                for key in ['TELESCOP', 'INSTRUME','EQUINOX','RA_TARG', 'DEC_TARG',
                            'PROPOSID', 'TDATEOBS',
                            'CCDAMP', 'CCDGAIN', 'CCDOFFST', 'OBSTYPE', 'OBSMODE',
                            'PHOTMODE', 'SUBARRAY', 'DETECTOR', 'OPT_ELEM',
                            'APERTURE', 'PROPAPER', 'FILTER', 'APER_FOV',
                            'CRSPLIT', 'PHOTZPT', 'PHOTPLAM', 'PHOTBW', ]:
                    try:
                        hdu.header[key] = (imgsHdrs[1][self.sciInds[0]][0][key],
                                           imgsHdrs[1][self.sciInds[0]][0].comments[key])
                    except:
                        print(f"Could not propagate header keyword {key}")
                        pass
                if self.bgCen is not None:
                    hdu.header['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
                    hdu.header['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
                else:
                    hdu.header['BGCENTY'] = (None, 'Science Y center background sample')
                    hdu.header['BGCENTX'] = (None, 'Science X center background sample')
                if self.bgCenRef is not None:
                    hdu.header['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
                    hdu.header['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
                else:
                    hdu.header['BGCENTYR'] = (None, 'Reference Y center background sample')
                    hdu.header['BGCENTXR'] = (None, 'Reference X center background sample')
                hdu.header['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
                hdu.header['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
                hdu.header['ORBCOMBI'] = (not noCombine, 'Stacked images per orbit before PSF sub?')
                hdu.header['ANNULI'] = (ann, 'Number of subtraction region annuli')
                hdu.header['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
                hdu.header['RADPROFS'] = (subRadProf, 'Residual radial profile subtracted?')
                if subFinalRadProf:
                    hdu.header.add_comment('Post-collapse radial profile subtracted')
                hdu.header.add_comment('{} PSF-subtracted combined image'.format(psfSubMode))
                hdu.header.add_comment('Reduction info file: {}'.format(infoPath))
                try:
                    hdu.header.add_comment(str(np.round(refScaleFactors, 5)).replace('\n', ''))
                except:
                    hdu.header.add_comment(str(refScaleFactors))
                hdu.header.add_comment(f'Constituent image exposure times (s): {exptimes_s}')
    
                hdu.writeto(dataDir + saveName, overwrite=True)
                print("\nPSF-subtracted combined image saved as {}".format(dataDir + saveName))
                
            if not self.noErrorMaps:
                print("\nComputing error maps...")
                sciHdrs = [imgsHdrs[1][ii] for ii in self.sciInds]
                rotImgs_electrons = convert_intensity(rotImgs.copy(), sciHdrs,
                                                      unitStart='counts s-1',
                                                      unitEnd='e-',
                                                      pscale=self.pscale)
                if targ not in noOffsetDatasets:
                    sourceMaskDerot = mask_exclusions(mask=spikemask,
                                           exclusions=exclusionsSci,
                                           cen=self.alignStar, cenOffset=self.alignStarOffsets[0],
                                           paOffset=0, spikeAngles=self.spikeAngles)
                else:
                    sourceMaskDerot = mask_exclusions(mask=spikemask,
                                           exclusions=exclusionsSci,
                                           cen=self.alignStar, cenOffset=np.zeros(2),
                                           paOffset=0, spikeAngles=self.spikeAngles)
                sourceMaskDerot[sourceMaskDerot == 1] = np.nan
                stdPACen = info['diskPA_deg'] - 90.
                if stdPACen < 0:
                    stdPACen += 360.
                stdPAMin = stdPACen - 40.
                if stdPAMin < 0:
                    stdPAMin += 360.
                    stdPARange1 = [stdPAMin, 360]
                else:
                    stdPARange1 = [stdPAMin, stdPACen]
                stdPAMax = stdPACen + 40.
                if stdPAMax > 360:
                    stdPAMax -= 360.
                    stdPARange2 = [0, stdPAMax]
                else:
                    stdPARange2 = [stdPACen, stdPAMax]
                if stdPAMin > stdPAMax:
                    stdPARange2 = [0, stdPAMax]
                stdPARange = stdPARange1 + stdPARange2
                errorMaps_electrons = []
    
                for ii in tqdm(range(len(rotImgs)), desc="Error maps"):
                    poissonMap = np.sqrt(np.abs(rotImgs_electrons[ii]))
                    theta = utils.make_phi(rotImgs_electrons[ii], self.alignStar,
                                           zeroAxis='+y')
                    # Get background noise map as std deviations of annuli.
                    stdMap = utils.get_partialann_stdmap(rotImgs_electrons[ii]+sourceMaskDerot,
                                        self.alignStar, radii, theta, stdPARange,
                                        r_max=400, rdelta=3)
                    # Extrapolate background noise into inner regions where we
                    # don't sample it well or at all but we do have data.
                    stdStrip1 = gaussian_filter(stdMap[self.alignStar[0], self.alignStar[1]-70:self.alignStar[1]], 3)
                    stdStrip2 = gaussian_filter(stdMap[self.alignStar[0], self.alignStar[1]+1:self.alignStar[1]+1+70], 3)
                    stdStrip = np.nanmean([stdStrip1, stdStrip2[::-1]], axis=0)
                    fe = interp1d(np.arange(self.alignStar[1]-70, self.alignStar[1])[~np.isnan(stdStrip)],
                                  gaussian_filter(stdMap[self.alignStar[0], self.alignStar[1]-70:self.alignStar[1]], 3)[~np.isnan(stdStrip)],
                                  kind='linear', fill_value="extrapolate")
                    interpStd = fe(np.arange(self.alignStar[1]-70, self.alignStar[1]))
                    for rr in range(1, 30): #len(interpStd)):
                        # if np.any(np.isnan(stdMap[(radii >= rr - 0.5) & (radii < rr + 0.5)])):
                        stdMap[(radii >= rr - 0.5) & (radii < rr + 0.5)] = interpStd[::-1][rr]
                    # Combine noise terms.
                    errorMaps_electrons.append(np.sqrt(np.nansum([poissonMap**2, stdMap**2], axis=0)))
                
                # Fractional error maps.
                fracerr = np.array(errorMaps_electrons)/rotImgs_electrons
                # Error maps in units of counts per second.
                errorMaps = fracerr*rotImgs # [counts s-1]
                # Number of non-NaN values in each pixel across all PSF-subtracted images.
                countAveragedPix = len(errorMaps) - np.sum(np.isnan(errorMaps), axis=0)
                
                # Propagated standard error of the mean.
                finalErrorMap = np.sqrt(np.nansum(errorMaps**2, axis=0))/np.sqrt(countAveragedPix)
                
                finalSNR = finalImg/finalErrorMap # [SNR]
                
                if saveFinal:
                    # Save the error map and SNR map too.
                    mapsToSave = [finalErrorMap, finalSNR]
                    filetypesToSave = ['Error map for final combined image', 'SNR map for final combined image']
                    stringsToSave = ['error_', 'snr_']
                    # if os.path.isfile(dataDir + saveName): raise IOError
                    for ii in range(len(mapsToSave)):
                        hdu = fits.PrimaryHDU(data=mapsToSave[ii].astype('float32'))
                        # hdu.header = data_hdr.copy()
                        if not 'HISTORY' in hdu.header:
                            try:                
                                hdu.header.add_history('Created by {}'.format(getpass.getuser()))
                            except:
                                hdu.header.add_history('Created by unknown user')
                        hdu.header['TARGNAME'] = (targ)
                        hdu.header['PSFNAME'] = (psfRefName, 'Reference PSF name')
                        hdu.header['FILETYPE'] = (filetypesToSave[ii])
                        hdu.header['NCOMBINE'] = (nSci, 'Number of images combined')
                        hdu.header['INPUTTYP'] = (suff, 'Type of input data')
                        hdu.header['PSFSUBMD'] = (psfSubMode, 'PSF-subtraction mode')
                        hdu.header['CENTRADN'] = (not self.noRadon, 'Radon transformed to get center?')
                        hdu.header['PSFCENTY'] = (self.alignStar[0], 'Y location of target star center')
                        hdu.header['PSFCENTX'] = (self.alignStar[1], 'X location of target star center')
                        hdu.header['ORIGCENY'] = (np.mean(self.starsOriginal, axis=0)[0], 'Un-padded mean star center Y')
                        hdu.header['ORIGCENX'] = (np.mean(self.starsOriginal, axis=0)[1], 'Un-padded mean star center X')
                        hdu.header['BUNIT'] = (bunit, 'brightness units')
                        hdu.header['BUNIT'] = (bunit, 'brightness units')
                        if self.bgCen is not None:
                            hdu.header['BGCENTY'] = (self.bgCen[0], 'Science Y center background sample')
                            hdu.header['BGCENTX'] = (self.bgCen[1], 'Science X center background sample')
                        else:
                            hdu.header['BGCENTY'] = (None, 'Science Y center background sample')
                            hdu.header['BGCENTX'] = (None, 'Science X center background sample')
                        if self.bgCenRef is not None:
                            hdu.header['BGCENTYR'] = (self.bgCenRef[0], 'Reference Y center background sample')
                            hdu.header['BGCENTXR'] = (self.bgCenRef[1], 'Reference X center background sample')
                        else:
                            hdu.header['BGCENTYR'] = (None, 'Reference Y center background sample')
                            hdu.header['BGCENTXR'] = (None, 'Reference X center background sample')
                        hdu.header['BGRADIUS'] = (self.bgRadius, 'Radius background sample region (pix)')
                        hdu.header['FIXPIX'] = (not self.noFixPix, 'Bad pixels were fixed?')
                        hdu.header['ORBCOMBI'] = (not noCombine, 'Stacked images per orbit before PSF sub?')
                        hdu.header['ANNULI'] = (ann, 'Number of subtraction region annuli')
                        hdu.header['SPWIDTH'] = (self.spWidth, 'Diff. spike mask width (pix)')
                        hdu.header['RADPROFS'] = (subRadProf, 'Residual radial profile subtracted?')
                        hdu.header.add_comment('{} PSF-subtracted combined image'.format(psfSubMode))
                        hdu.header.add_comment('Reduction info file: {}'.format(infoPath))
                        
                        hdu.writeto(dataDir + stringsToSave[ii] + saveName, overwrite=True)
                        print("\n{} saved as {}".format(filetypesToSave[ii], dataDir + stringsToSave[ii] + saveName))
    
    # # TEMP!!! Plot PSF-subtracted, derotated images.
    #     fig = plt.figure(1)
    #     for ii in range(len(rotImgs)):
    #         st = np.round(stars[self.sciInds][ii]).astype(int)
    #         fig.clf()
    #         ax = plt.subplot(111)
    #         patchYLims = st[0] + roiY
    #         patchXLims = st[1] + roiX
    #         patchYLims[patchYLims < 0] = 0
    #         patchXLims[patchXLims < 0] = 0
    #         ax.imshow(rotImgs[ii][patchYLims[0]:patchYLims[1], patchXLims[0]:patchXLims[1]],
    #                   norm=SymLogNorm(linthresh=1., linscale=1., vmin=-5., vmax=60.))
    #         ax.set_title("Derotated, PSF-subtracted Image {}".format(ii))
    #         plt.draw()
    #         pdb.set_trace()
    #     
    #     fig = plt.figure(10)
    #     fig.clf()
    #     ax = plt.subplot(111)
    #     patchFinal = finalImg.copy()[radii <= 40]
    #     vminFinal = np.percentile(np.nan_to_num(patchFinal,0), 1.)
    #     vmaxFinal = np.percentile(np.nan_to_num(patchFinal,0), 99.99)
    #     ax.imshow(finalImg,
    #               norm=SymLogNorm(linthresh=0.1, linscale=1., vmin=vminFinal, vmax=vmaxFinal))
    #     ax.set_title("PSF-subtracted, combined image")
    #     plt.draw()
        
    
        print("\nFin")
