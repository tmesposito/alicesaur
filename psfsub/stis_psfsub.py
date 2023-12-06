#!/usr/bin/env python
#
# Tom Esposito

import os
import pdb
import ctypes
import numpy as np
from numpy.polynomial.polynomial import polyval2d
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from scipy.ndimage import gaussian_filter, shift, filters
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from tqdm import tqdm

# Internal imports
from alicesaur.utils import *
from alicesaur.plot.disk_plot import measure_radial_profile


def measure_mean_radial_prof(img, cen, paList=[0., 90., 180., 270.], paHW=None,
                             rMax=180, interpInf=True, expandHW_r=False,
                             expandHW=None):
    """
    paList: list, position angles east of +y axis in [degrees].
    paHW: half-width of PA wedge to include on either side of pa [deg].
    rMax: max radius of profile to measure [pixels].
    
    Output:
        Smoothed 2D map of the median radial profile measured at paList.
    """

    profList = []
    radList = []

    for pa in paList:
        rads, prof, profOpp, paPeak, paOppPeak = measure_radial_profile(img,
                                    star=cen, pa=pa, rMax=rMax, mode='mean',
                                    paHW=paHW, height=None, plot=False,
                                    expandHW_r=expandHW_r, expandHW=expandHW)
        # Clean the profiles of high outliers.
        prof = np.array(prof)
        # profOpp = np.array(profOpp)
        # prof[prof > 5*np.nanmedian(prof)] = np.nanmedian(prof)
        # profOpp[profOpp > 5*np.nanmedian(profOpp)] = np.nanmedian(profOpp)
        profList.append(prof)
        radList.append(rads)

    meanRads = np.mean(radList, axis=0)
    medProf = np.nanmedian(profList, axis=0)
    interped = None

    if interpInf:
        firstGoodInd = 0
        # Find first good measurement and start there.
        for ii, val in enumerate(medProf):
            if not np.isnan(medProf[ii]):
                firstGoodInd = ii
                break
        # Also find the last good measurement so don't try to interpolate past it later.
        if firstGoodInd == len(medProf) - 1:
            lastGoodInd = firstGoodInd
        else:
            lastGoodInd = firstGoodInd
            for ii in range(firstGoodInd, len(medProf)):
                lastGoodInd += 1
                if np.isnan(medProf[ii]):
                    if np.all(np.isnan(medProf[ii:])):
                        lastGoodInd = ii - 1
                        break
            if lastGoodInd == len(medProf):
                lastGoodInd -= 1

        # Interpolate over radii with np.nan or +/-np.inf as the value.
        # if ii < len(medProf) - 1:
        if lastGoodInd > firstGoodInd + 2:
            wh_isNan = np.isnan(medProf[firstGoodInd:lastGoodInd+1]) | np.isinf(medProf[firstGoodInd:lastGoodInd+1])
            func = interp1d(meanRads[firstGoodInd:lastGoodInd+1][~wh_isNan], medProf[firstGoodInd:lastGoodInd+1][~wh_isNan])
            new = func(meanRads[firstGoodInd:lastGoodInd+1])
            interped = medProf.copy()
            interped[firstGoodInd:lastGoodInd+1][wh_isNan] = new[wh_isNan]

    # Make a 2D radial profile map.
    radii = make_radii(img, cen)
    radProf2d = np.zeros(img.shape)
    for ii, rad in enumerate(meanRads):
        if interped is not None:
            radProf2d[(radii >= rad - 0.5) & (radii < rad + 0.5)] = interped[ii]
        else:
            radProf2d[(radii >= rad - 0.5) & (radii < rad + 0.5)] = medProf[ii]

    # Smooth the 2D radial profile map; then fix the NaNs near the center.
    radProf2d_smooth = gaussian_filter(radProf2d, 2.)
    radProf2d_smooth[np.isnan(radProf2d_smooth)] = radProf2d[np.isnan(radProf2d_smooth)]

    # plt.figure(4)
    # plt.clf()
    # plt.imshow(radProf2d_smooth, vmin=-20., vmax=20.)
    # plt.draw()
    # 
    # pdb.set_trace()

    return radProf2d_smooth


def dither_image(im, star, ditherPos):
    """
    Shift an image by interpolation; primarily within a dither pattern.

    Parameters
    ----------
    ditherPos: array
        [dy, dx] shifts of the image center.
    """

    imDithered = shift(im, ditherPos, order=1, cval=0.)

    return imDithered


def dither_subtract_psf(sci, ref, refStar, shift=0.01, nIm=0):
    """
    Parameters
    ----------
    sci : array
        Science image to subtracted the PSF from.
    ref : array
        Reference PSF image to subtract from sci.
    refStar : array
        Star y,x positions for the reference image.
    shift : float, optional
        Number of pixels by which to dither the reference. The default is 0.5.

    Returns
    -------
    None.

    """
    
    # 9-point dither, starting with up+left, then going across the row to
    # up+right, then ending with down+right. Middle (no dither) is included.
    ditherShifts = np.array([[shift, -shift], [shift, 0], [shift, shift],
                             [0, -shift], [0, 0], [0, shift],
                             [-shift, -shift], [-shift, 0], [-shift, shift]])
    
    # Original chi2, with no dither.
    chi2_orig = np.nansum((sci - ref)**2)
    
    print("\nTesting 9-point dither for better PSF subtraction...\n")
    
    ditheredSubs = []
    chi2Subs = []
    for ii, pos in enumerate(ditherShifts):
        
        if np.all(pos == np.array([0,0])):
            chi2Subs.append(np.inf)
            ditheredSubs.append(sci - ref)
            continue
        
        # Clean NaN's from ref before dither, then replace them after dither.
        refDithered = dither_image(ref, refStar, pos)
        res = sci - refDithered
        chi2 = np.nansum(res**2)
        ditheredSubs.append(res)
        chi2Subs.append(chi2)
    
    print(chi2Subs)
    whMinChi2 = np.argmin(np.array(chi2Subs))
    print(f"\nOriginal chi^2 (no dither) = {chi2_orig:.2f}")
    print(f"Minimum dithered chi^2     = {chi2Subs[whMinChi2]:.2f}")
    print(f"from dither position {whMinChi2}: {ditherShifts[whMinChi2]}\n")
    
    # plt.figure(1)
    # plt.clf()
    # plt.title('Original')
    # plt.imshow(sci - ref, vmin=-1, vmax=1)
    # plt.draw()
    
    # plt.figure(2)
    # plt.clf()
    # plt.title(f'Dither {whMinChi2} ({ditherShifts[whMinChi2]})')
    # plt.imshow(ditheredSubs[whMinChi2][int(refStar[0]-150):int(refStar[0]+150),
    #                                    int(refStar[1]-150):int(refStar[1]+150)],
    #            vmin=-1, vmax=1)
    # plt.draw()
    
    fig, axs = plt.subplots(3,3)
    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, wspace=0, hspace=0)
    for ii, pos in enumerate(ditherShifts):
        ax = axs.flatten()[ii]
        ax.imshow(ditheredSubs[ii][int(refStar[0]-150):int(refStar[0]+150),
                                   int(refStar[1]-150):int(refStar[1]+150)],
                  vmin=-1, vmax=1)
        if ii in range(0,3):
            ax.set_title(f'Dither {ii} ({ditherShifts[ii][0]}, {ditherShifts[ii][1]})',
                         fontsize=12)
        if np.all(pos == np.array([0,0])):
            ax.text(0.5, 0.9, f'{chi2_orig:.2f}', c='k', 
                    horizontalalignment='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.9, f'{chi2Subs[ii]:.2f}', c='k', 
                    horizontalalignment='center', transform=ax.transAxes)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    plt.draw()
    fig.savefig(f'dither_PSF_sub_image_{nIm}.png', format='png')
    
    if chi2Subs[whMinChi2] < chi2_orig:
        bestDither = ditherShifts[whMinChi2]
        bestRef = sci - ditheredSubs[whMinChi2]
    else:
        bestDither = np.array([0,0])
        bestRef = ref

    return bestDither, bestRef


def dither_residuals(ps, sci, ref, refMask, refStar, verbose=False):
    """
    Parameters
    ----------
    sci : array
        Science image to subtracted the PSF from.
    ref : array
        Reference PSF image to subtract from sci.
    refStar : array
        Star y,x positions for the reference image.
    shift : float, optional
        Number of pixels by which to dither the reference. The default is 0.5.

    Returns
    -------
    None.

    """

    # Original chi2, with no dither.
    chi2_orig = np.nansum((sci - ref)**2)

    pos = np.array([ps[0], ps[1]])

    # Clean NaN's from ref before dither, then replace them after dither.
    refDithered = dither_image(ref, refStar, pos)
    refDitheredMasked = refDithered.copy()
    refDitheredMasked[refMask] = np.nan
    res = sci - refDitheredMasked
    chi2 = np.nansum(res**2)

    if verbose:
        print(f"\nOriginal chi^2  = {chi2_orig:.2f}")
        print(f"Iteration chi^2 = {chi2:.2f}  (dither: {pos})\n")

# # TEMP!!!
#     plt.figure(1)
#     plt.clf()
#     plt.title(f'Original chi2={chi2_orig:.2f}')
#     plt.imshow((sci - ref)[int(refStar[0]-150):int(refStar[0]+150),
#                             int(refStar[1]-150):int(refStar[1]+150)],
#                 vmin=-1, vmax=1)
#     plt.draw()

#     plt.figure(2)
#     plt.clf()
#     plt.title(f'Dither ({pos[0]:.4f}, {pos[1]:.4f}) chi2={chi2:.2f}')
#     plt.imshow(res[int(refStar[0]-150):int(refStar[0]+150),
#                     int(refStar[1]-150):int(refStar[1]+150)],
#                 vmin=-1, vmax=1)
#     plt.draw()
    
#     pdb.set_trace()
    
    return np.extract(~np.isnan(res), res)


def rdi_subtract_psf(sciImgs, refImgs, sciMasks, refMasks, sciStars,
                     C0=0., rmin=0, rmax=None, ann=1, orientats=None,
                     radProfPaList=np.array([0.]), radProfPaHW=40,
                     radProfMax=200, radProfMasks=None,
                     bgCen=None, bgRadius=30, subRadProf=True,
                     optimize_dither=True):
    """
        C0: initial guess for log10 of scalar multiplier of ref PSF.
        subRadProf: bool, True to subtract a radial profile after main PSF subtraction.
        optimize_dither: bool, True to optimize PSF subtraction by running a
            least-squares to minimize PSF residuals while dithering the
            reference image in X and Y. False to skip this.
    """
    
    def residuals_multi_ref(ps, sci, refs, weightMap=1.):
        weights = 10**ps
        resMasked = weightMap*(sci - np.sum((weights.transpose()*refs.transpose()).transpose(), axis=0))

        res = np.extract(~np.isnan(resMasked), resMasked)

        # !!!! IMPORTANT NOTE !!! leastsq takes the 1st order residuals (data - model),
        # and NOT the 2nd order residuals ( (data - model)**2 ).
        # Using the latter will cause it to give poor results.
        return res
        # Try to ignore bright residuals from bg stars.
        # return res[(res >= p5) & (res <= p95)]
        # return res[res <= p99]

    def residuals(ps, sci, ref, weightMap=1., debug=False):
        if debug:
            print(ps[0],
                  np.sum(np.nan_to_num(sci - (10**ps[0])*ref, 0.).flatten()**2))
            plt.figure(12)
            plt.clf()
            plt.imshow(sci, vmin=-1., vmax=1000.)
            plt.title("'sci' = Science image")
            plt.figure(13)
            plt.clf()
            plt.imshow(ref, vmin=-1., vmax=1000.)
            plt.title("'ref' = input Reference image")
            plt.figure(14)
            plt.clf()
            plt.imshow(sci - (10**ps[0])*ref, vmin=-1., vmax=10.)
            plt.draw()
            plt.title("Residuals = 'sci' - scaled 'ref'")
            plt.figure(15)
            plt.clf()
            plt.imshow((sci - (10**ps[0])*ref)/weightMap, vmin=-1., vmax=10.)
            plt.draw()
            plt.title("Weighted Residuals = ('sci' - scaled 'ref')/weights")
            pdb.set_trace()

        return np.nan_to_num((sci - (10**ps[0])*ref)*weightMap, 0.).flatten()
    
    def residuals_poly(ps, sci, ref, yy, xx, rr, k):
        if k > 0:
            # poly = ps[1]*yy**2 + ps[2]*xx**2 + ps[3]*yy*xx + ps[4]*yy + ps[5]*xx + ps[6]
            # cs = ps.reshape((k+1, k+1))
            # poly = polyval2d(xx, yy, cs)
            # poly = ps[0]*(5**0.5 * (6*rr**4 - 6*rr**2 + 1)) # spherical Zernike
            poly = ps[0]*(ps[1] + ps[2]*rr**2 + ps[3]*rr**4) # polynomial
            res = np.nan_to_num(sci - poly, 0.).flatten() # residuals
            print(ps)
            print(np.sum(res**2)/(np.sum(~np.isnan(res))-4.)) # reduced chi2
            # plt.figure(1)
            # plt.clf()
            # plt.imshow(poly)
            # plt.draw()
            # plt.figure(2)
            # plt.clf()
            # plt.imshow(sci - poly, vmin=-1, vmax=1)
            # plt.draw()
            
            # breakpoint()
            # return np.nan_to_num(sci - (10**ps[0])*ref - poly, 0.).flatten()
            return res
        else:
            return np.nan_to_num(sci - (10**ps[0])*ref, 0.).flatten()

    subImgs = []
    refScaleFactors = []
    radii = make_radii(sciImgs[0], sciStars[0])
    if rmax is None:
        rmax = np.nanmax(radii)
    if len(refImgs) == 1:
        p0 = np.array([C0])
        refImgMasked = refImgs[0].copy()
        refImgMasked[refMasks[0]] = np.nan
        for ii in range(len(sciImgs)):
            img = sciImgs[ii].copy()
            img[sciMasks[ii]] = np.nan
            sciStar = sciStars[ii]
            # weightMap = get_ann_stdmap(img, sciStars[ii], radii, r_max=rmax+3,
            #                            use_median=True)
            # weightMap *= radii
            weightMap = 1.
            # For annular subsections.
            if ann > 1:
                rstep = (rmax - rmin)/float(ann)
                subImg = sciImgs[ii].copy()
                annRinList = np.arange(rmin, rmax + 1, rstep)
                for jj in range(len(annRinList) - 1):
                    annulus = (radii >= annRinList[jj]) & (radii < annRinList[jj+1])
                    imgAnnulus = img.copy()
                    imgAnnulus[~annulus] = np.nan
                    pf = leastsq(residuals, p0, args=(imgAnnulus, refImgMasked))
                    subImg[annulus] = (sciImgs[ii] - (10**pf[0])*refImgMasked)[annulus]
            # For full image (no subsection) subtraction.
            else:
                k = 0 # degree of polynomial
                yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
                yy = yy - sciStar[0]
                xx = xx - sciStar[1]
                rho = radii.copy()
                rho[radii > 100] = np.nan
                rho /= np.nanmax(rho)
                c0 = np.array([1., -1., 2., -2.])
                # c0 = np.zeros((k+1, k+1))
                # p0 = np.append(np.array([0.]), c0.flatten())
                pf = leastsq(residuals, p0, args=(img, refImgMasked, weightMap),
                             full_output=1, factor=1, epsfcn=0.01)
                # pf = leastsq(residuals_poly, p0, args=(img, refImgMasked, yy, xx, k))
                if k > 0:
                    rads, prof, profOpp, paPeak, paOppPeak = measure_radial_profile(img - (10**pf[0][0])*refImgMasked,
                                                star=sciStar, pa=-30.,
                                                height=None)
                    # Clean the profiles of high outliers.
                    prof = np.array(prof)
                    profOpp = np.array(profOpp)
                    prof[prof > 5*np.nanmedian(prof)] = np.nanmedian(prof)
                    profOpp[profOpp > 5*np.nanmedian(profOpp)] = np.nanmedian(profOpp)
                    # Fit polynomial of degree pdeg to 1d profiles.
                    pdeg = 7
                    Cprof = np.polyfit(rads[(rads <= 200) & (~np.isnan(prof))], prof[(rads <= 200) & (~np.isnan(prof))], pdeg)
                    CprofOpp = np.polyfit(rads[(rads <= 200) & (~np.isnan(profOpp))], profOpp[(rads <= 200) & (~np.isnan(profOpp))], pdeg)
                    p1prof = np.poly1d(Cprof)(rads)
                    p1profOpp = np.poly1d(CprofOpp)(rads)
                    plt.figure(19)
                    plt.clf()
                    plt.imshow(img - (10**pf[0][0])*refImgMasked, vmin=-1, vmax=1)
                    plt.draw()
                    plt.figure(20)
                    plt.clf()
                    plt.plot(rads, prof)
                    plt.plot(rads, profOpp)
                    plt.plot(rads, p1prof, 'c--')
                    plt.plot(rads, p1profOpp, 'm--')
                    plt.draw()

                    poly_out = leastsq(residuals_poly, c0.flatten(),
                                 args=(img - (10**pf[0][0])*refImgMasked, refImgMasked, yy, xx, rho, k),
                                 epsfcn=0.01,
                                 factor=5,
                                 full_output=1)
                    pfpoly = poly_out[0]
                    try:
                        print(poly_out[3])
                    except:
                        pass
                    # cf = pfpoly[0].reshape((k+1, k+1))
                    # polybf = polyval2d(xx, yy, cf)
                    polybf = pfpoly[0]*(pfpoly[1]*rho**4 - pfpoly[2]*rho**2 + pfpoly[3]) # spherical Zernike
                    subImg = sciImgs[ii] - (10**pf[0][0])*refImgMasked - polybf
                    print(pfpoly)
                else:
                    subImg = sciImgs[ii] - (10**pf[0])*refImgMasked
                    refScaleFactors.append(10**pf[0][0])

                    if optimize_dither:
                        p0_dither = np.array([0.1, 0.1])
                        pf_dither = leastsq(dither_residuals, p0_dither,
                                     args=(img, (10**pf[0][0])*refImgs[0].copy(),
                                           refMasks[0], np.array([1024., 1024.]),
                                           True),
                                     full_output=1, factor=1, epsfcn=0.1)
                        bestDither = np.array([pf_dither[0][0], pf_dither[0][1]])

                        # Replace the previous best subtraction with the better one
                        if not np.all(bestDither == np.array([0,0])):
                            ditheredRef = dither_image((10**pf[0][0])*refImgs[0],
                                                       star=np.array([1024., 1024.]),
                                                       ditherPos=bestDither)
                            ditheredRefMasked = ditheredRef.copy()
                            ditheredRefMasked[refMasks[0]] = np.nan
                            subImg = sciImgs[ii] - ditheredRefMasked
                            print(f"Updated PSF subtraction with best dither "\
                                  f"{bestDither}")

            # Optionally subtract the distant background again.
            if bgCen is not None:
                bgCenRot = rotate_yx(bgCen, sciStar, orientats[ii])
                subImg, bg = subtract_bg(subImg, bgCenRot, bgRadius)
            # Measure and subtract a radial profile from the
            # PSF-subtracted individual image.
            if subRadProf:
                subImgMasked = subImg.copy()
                if radProfMasks is not None:
                    subImgMasked[radProfMasks[ii]] = np.nan
                meanRadProf = measure_mean_radial_prof(subImgMasked, sciStar,
                                                       paList=radProfPaList - orientats[ii],
                                                       paHW=radProfPaHW, rMax=radProfMax,
                                                       interpInf=True)
                meanRadProf = np.nan_to_num(meanRadProf, 0)
                subImgs.append(subImg - meanRadProf)
            else:
                subImgs.append(subImg)
                meanRadProf = 0
            # plt.figure(13)
            # plt.clf()
            # plt.imshow(subImg - meanRadProf, vmin=-1., vmax=10.)
            # plt.draw()
            # pdb.set_trace()
            # print(pf)
                
            # plt.figure(1)
            # plt.clf()
            # plt.imshow(polybf)
            # plt.draw()
            # plt.figure(2)
            # plt.clf()
            # plt.imshow(sciImgs[ii]- (10**pf[0][0])*refImgMasked - polybf, vmin=-1, vmax=1)
            # plt.draw()
            # breakpoint()

# *** MULTI-REFERENCE IMAGE CASE ***
    else:
        p0 = np.tile(C0, len(refImgs))
        # Mask all reference images.
        refImgsMasked = refImgs.copy()
        for ii, rIm in enumerate(refImgsMasked):
            refImgsMasked[ii][refMasks[ii]] = np.nan
        # Do the PSF subtraction on each science image.
        for ii in tqdm(range(len(sciImgs))):
            img = sciImgs[ii].copy()
            img[sciMasks[ii]] = np.nan
            sciStar = sciStars[ii]
            # weightMap = get_ann_stdmap(img, sciStars[ii].astype(int), radii,
            #                            r_max=rmax+3, use_median=True)
            # weightMap *= radii
            weightMap = 1.
            # For annular subsections.
 # FIX ME!!! Multiple annuli not tested for multi refs.
            if ann > 1:
                if rmax is None:
                    rmax = np.nanmax(radii)
                rstep = (rmax - rmin)/float(ann)
                subImg = sciImgs[ii].copy()
                annRinList = np.arange(rmin, rmax + 1, rstep)
                for jj in range(len(annRinList) - 1):
                    annulus = (radii >= annRinList[jj]) & (radii < annRinList[jj+1])
                    imgAnnulus = img.copy()
                    imgAnnulus[~annulus] = np.nan
                    pf = leastsq(residuals, p0, args=(imgAnnulus, refImgMasked))
                    subImg[annulus] = (sciImgs[ii] - (10**pf[0])*refImgMasked)[annulus]
            # For full image (no subsection) subtraction.
            else:
                k = 0 # degree of polynomial
                pf = leastsq(residuals_multi_ref, p0, args=(img,
                                                    refImgsMasked, weightMap),
                              full_output=1, factor=1, epsfcn=0.1)
                if k > 0:
                    print("\n***HELP!!! Poly fit not implemented for "\
                          "multiple ref images. Sorry!\n")
                    pass
                else:
                    subImg = sciImgs[ii] - np.sum(((10**pf[0].transpose())*refImgsMasked.transpose()).transpose(), axis=0)
                    refScaleFactors.append(10**pf[0])

                if optimize_dither:
                    p0_dither = np.array([0.1, 0.1])
                    pf_dither = leastsq(dither_residuals, p0_dither,
                                 args=(img,
                                       np.sum(((10**pf[0].transpose())*refImgs.transpose()).transpose(), axis=0),
                                       refMasks[0], np.array([1024., 1024.]),
                                       True),
                                 full_output=1, factor=1, epsfcn=0.1)
                    bestDither = np.array([pf_dither[0][0], pf_dither[0][1]])

                    # Replace the previous best subtraction with the better one.
                    if not np.all(bestDither == np.array([0,0])):
                        ditheredRef = dither_image(np.sum(((10**pf[0].transpose())*refImgs.transpose()).transpose(), axis=0),
                                                   star=np.array([1024., 1024.]),
                                                   ditherPos=bestDither)
                        ditheredRefMasked = ditheredRef.copy()
                        ditheredRefMasked[refMasks[0]] = np.nan
                        subImg = sciImgs[ii] - ditheredRefMasked
                        print(f"Updated PSF subtraction with best dither "\
                              f"{bestDither}")

            # Optionally subtract the distant background again.
            if bgCen is not None:
                bgCenRot = rotate_yx(bgCen, sciStar, orientats[ii])
                subImg, bg = subtract_bg(subImg, bgCenRot, bgRadius)
            # Measure and subtract a radial profile from the
            # PSF-subtracted individual image.
            if subRadProf:
                subImgMasked = subImg.copy()
                if radProfMasks is not None:
                    subImgMasked[radProfMasks[ii]] = np.nan
                meanRadProf = measure_mean_radial_prof(subImgMasked, sciStar,
                                        paList=radProfPaList - orientats[ii],
                                        paHW=radProfPaHW, rMax=radProfMax)
                meanRadProf = np.nan_to_num(meanRadProf, 0)
                subImgs.append(subImg - meanRadProf)
                print("\nSubtracted mean radial profile after main "\
                      "PSF subtraction\n")
            else:
                subImgs.append(subImg)
                meanRadProf = 0

            print(pf)

        subImgs = np.array(subImgs)

    return subImgs, refScaleFactors


def adi_subtract_psf():
    
    pass

    return


def make_stis_dataset(dataset, inputPaths, inputImgs=None, IWA=None, OWA=None,
                      star=None, parangs=None, aligned_center=None):

    if inputImgs is not None:
        inputData = inputImgs
    else:
        inputData = []
    inputCenters = []
    PAs = []
    prihdrs = []
    # Select the correct data array from the HDU.
    for ii, ff in enumerate(inputPaths):
        with fits.open(ff, mode='readonly') as hdul:
            hdr0 = hdul[0].header
            hdr1 = hdul[1].header
            prihdrs.append(hdr0)
            if inputImgs is None:
                inputData.append(hdul[1].data)
            if star is not None:
                inputCenters.append(star[ii])
            else:
                ww = wcs.WCS(hdr1)
                targRA = hdr0['RA_TARG']
                targDec = hdr0['DEC_TARG']
                inputCenters.append(ww.wcs_world2pix([[targRA, targDec]], 0)[0][::-1]) # [pixels] y,x
            if parangs is not None:
                PAs.append(parangs[ii])
            else:
                PAs.append(hdr1["ORIENTAT"]) # [degrees]

    dataset.input = np.array(inputData)
    dataset.centers = np.array(inputCenters)[::, ::-1] # flip to X,Y
    dataset.PAs = np.array(PAs)
    dataset.prihdrs = prihdrs
    dataset.IWA = IWA
    dataset.OWA = OWA
    dataset.filenames = np.array(inputPaths)
    dataset.flipx = False

    # Fudge some dataset parameters.
    dataset.wcs = [None]
    dataset.wvs = np.ones(len(dataset.input))
    dataset.filenums = np.arange(float(len(dataset.input)))
    dataset.lenslet_scale = 0.0507 # [arcsec/pixel]

    if np.nan in dataset.PAs:
        print("WARNING: No PA found for one or more images, pyKLIP will probably crash.")
    if np.nan in dataset.centers:
        print("WARNING: No CENTER found for one or more images, pyKLIP will probably crash.")

    # Get mean X,Y star position for diffraction spike masking.
    star_mean = np.mean(dataset.centers, axis=0)
    # Set star center for aligned images and final KLIP output.
    if aligned_center is not None:
        dataset.aligned_center = aligned_center
    else:
        dataset.aligned_center = np.round(star_mean)
    
    return dataset


# FIX ME!!! do_klip_stis has too many private dependencies. Won't import as is.
# def do_klip_stis(ds, inputPaths, inputImgs=None, inputHdrs=None,
#                 psfPaths=None, psfImgs=None, mode='ADI', 
#                 ann=1, subs=1, minrot=0, mvmt=0, IWA=10., OWA=400.,
#                 numbasis=[1,10,50], maxnumbasis=None, star=None, highpass=False,
#                 pre_sm=None, spWidth=0., ps_spWidth=0., PAadj=0.,
#                 parangs=None, aligned_center=None, collapse="mean", prfx="",
#                 save_psf_cubes=False, save_aligned=False, restored_aligned=None,
#                 lite=True, do_snmap=False, numthreads=8, sufx='', output=False,
#                 compute_correlation=True):
#     """
#     Run pyKLIP on HST/STIS images. Use Jason Wang's parallelized lite pyKLIP.
    
#     Inputs:
#         ds: str dataset name
#         inputPaths: list of paths to input image fits files.
#         inputImgs: optional list of arrays of input images.
#         psfPaths: optional list of paths to reference PSF image fits files.
#         psfImgs: optional list of arrays of reference PSF images.
#         ann: number of annuli to use for KLIP
#         subs: number of sections to break each annulus into
#         minrot: minimum PA rotation (in degrees) to be considered for use as a
#             reference PSF (good for disks). 0 if no constraint (default).
#         mvmt: minimum amount of movement (in pixels) of an astrophysical source
#             to consider using that image for a reference PSF
#         IWA: inner working angle (in pixels)
#         OWA: outer working angle (in pixels)
#         numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
#         maxnumbasis:
#         star: 
#         fsc:
#         parangs: optional pa_list for the science frames (not ref frames) [deg]
#         numthreads: number of threads to use. If None, defaults to using all the cores of the cpu
#         ##maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
#         pre_sm: size of smoothing filter applied to all images before starting KLIP [pix]
#         sufx: str suffix at end of filename.
#         output: if True, function returns mean-collapsed cube. Otherwise, no output [bool].
        
#     Outputs:
#         KLIP'd cube, rot_cube, and collapsed images all saved to disk.
#         If output=True, returns mean-collapsed cube. Otherwise, no function output.
#     """
#     from ttools import getBasics, rot_array, get_spikemask
#     from calibrate import headerWrite
#     # import pyfits # for some reason astropy.io.fits causes header END card issues.
#     from pyklip import rdi
#     from pyklip.instruments import NIRC2
#     from pyklip.parallelized import klip_dataset
    
#     if highpass:
#         print("Highpass filtering images after loading:", highpass)
    
# # # TEMP!!! Start with an empty NIRC2Data object.
# #     example = NIRC2.NIRC2Data(['/Users/Tom/Research/data/hd191089_160625_H/processed/cn0274.fits',
# #                                '/Users/Tom/Research/data/hd191089_160625_H/processed/cn0275.fits',
# #                                '/Users/Tom/Research/data/hd191089_160625_H/processed/cn0276.fits'])

#     # # TEMP!!! Hack in some images
#     # inputPaths = sorted(['/Users/Tom/Research/data/hst/hd129590_20200413_stis/bar10/odxy17010_sx2.fits.gz',
#     #                     '/Users/Tom/Research/data/hst/hd129590_20200413_stis/bar10/odxy19010_sx2.fits.gz',
#     #                     '/Users/Tom/Research/data/hst/hd129590_20200413_stis/bar10/odxy20010_sx2.fits.gz'])
#     # 
#     # # psflib_filenames must include inp_filename list to work.
#     # psfPaths = inputPaths + sorted(['/Users/Tom/Research/data/hst/hd129590_20200413_stis/bar10/odxy18010_sx2.fits.gz',
#     #                            '/Users/Tom/Research/data/hst/hd117214_20200517_stis/bar10/odxy14010_sx2.fits.gz',
#     #                            '/Users/Tom/Research/data/hst/hd115600_20200210_stis/bar10/odxy10010_sx2.fits.gz'])


#     # Construct dataset. Make one for the RDI PSF library too, if necessary.
#     dataset = NIRC2.NIRC2Data()
#     dataset = make_stis_dataset(dataset, inputPaths, inputImgs=inputImgs,
#                                 IWA=IWA, OWA=OWA, aligned_center=aligned_center,
#                                 star=star, parangs=parangs)

#     if "RDI" in mode.upper():
#         dataset_psflib = NIRC2.NIRC2Data()
#         dataset_psflib = make_stis_dataset(dataset_psflib, psfPaths, inputImgs=psfImgs,
#                                     IWA=IWA, OWA=OWA, aligned_center=aligned_center,
#                                     star=star, parangs=None)
#         lite = False # force non-lite memory mode (required by RDI)
#         nPSFs = str(len(dataset_psflib.filenames) - len(dataset.filenames))
#     else:
#         nPSFs = ''

#     outputDir = os.path.dirname(dataset.filenames[0])

#     if mvmt is not None:
#         fileprefix = "%sa%ds%dmv%d_hp%s_%s%s_k%d-%d" % (prfx, ann, subs, mvmt, str(highpass), mode, nPSFs, numbasis[0], numbasis[-1])
#     elif minrot is not None:
#         fileprefix = "%sa%ds%dmr%d_hp%s_%s%s_k%d-%d" % (prfx, ann, subs, minrot, str(highpass), mode, nPSFs, numbasis[0], numbasis[-1])


#     # # High-pass filter (unsharp mask) the pre-KLIP images if desired.
#     # if (fsc != 0.) & (hdr['FILETYPE'] in ['radial sub cube', 'aligned image cube']):
#     #     data = unsharp(ds, hdu, fl, fsc, ident='', silent=True)
#     # elif hdr['FILETYPE'] == 'unsharp mask cube':
#     #     print "Using pre-filtered cube (sigma=%.1f pix) as input data..." % fsc
#     # else:
#     #     print "Using unfiltered cube as input data..."

#     # Smooth the individual images w/ Gaussian before doing anything with them.
#     if pre_sm is not None:
#         print("Smoothing pre-KLIP data with Gaussian of sigma=%.2f pix" % pre_sm)
#         dataset.input = gaussian_filter(dataset.input.copy(), pre_sm)
#         # data = filters.median_filter(data, pre_sm)

#     # Mask diffraction spikes at fixed angles relative to detector frame.
#     print("Masking diffraction spikes pre-KLIP...")
#     masks = []
#     # Angles at which diffraction spikes occur in STIS data [deg].
#     spikeAngles = np.array([45.0, 135.0]) # [deg] w/ 0 at +X
    
#     if spWidth > 0:
#         for ii, img in enumerate(dataset.input):
#             spikemask = make_spikemask_stis(img, dataset.centers[ii][::-1],
#                                             spikeAngles, width=spWidth)
#     # FIX ME!!! Combine spikemask with occulter mask here.
#             masks.append(spikemask)
#         # Apply spikemasks to data.
#         for ii, img in enumerate(dataset.input):
#             dataset.input[ii] = np.ma.array(dataset.input[ii].copy(), mask=masks[ii]).filled(np.nan)


#     # For RDI, compute the correlation matrix for all reference and
#     # science frames, or load a pre-computed matrix.
#     if "RDI" in mode.upper():
#         psflib_imgs = dataset_psflib.input
#         if compute_correlation:
#             psflib = rdi.PSFLibrary(psflib_imgs, dataset.aligned_center, dataset_psflib.filenames,
#                                     compute_correlation=True)
#             psflib.save_correlation(os.path.join(outputDir, "%s_corr_matrix.fits" % (fileprefix)),
#                                     overwrite=True)
#         else:
#             # To load in a saved correlation matrix:
#             corr_matrix_hdulist = fits.open(os.path.join(outputDir, "%s_corr_matrix.fits" % (fileprefix)))
#             corr_matrix = corr_matrix_hdulist[0].data
#             psflib = rdi.PSFLibrary(psflib_imgs, dataset.aligned_center, dataset_psflib.filenames,
#                                     compute_correlation=False,
#                                     correlation_matrix=corr_matrix)
    
#         psflib.prepare_library(dataset)
#         psflib.master_wvs = np.ones(psflib.master_library.shape[0])
#     else:
#         psflib = None

#     # Do the KLIP subtraction.
#     klip_dataset(dataset, mode=mode, outputdir=outputDir, fileprefix=fileprefix,
#                 annuli=ann, subsections=subs, movement=mvmt, minrot=minrot,
#                 numbasis=numbasis, maxnumbasis=maxnumbasis, numthreads=numthreads,
#                 annuli_spacing="constant", aligned_center=dataset.aligned_center, 
#                 algo="klip", calibrate_flux=False, lite=lite, time_collapse=collapse,
#                 highpass=False, psf_library=psflib, dtype=ctypes.c_float,
#                 save_aligned=save_aligned, restored_aligned=restored_aligned)


#  #    try:
#  #        os.path.exists(save)
#  #        save_hdu = fits.PrimaryHDU(data=mean_arr.astype('float32'))
#  #        save_hdu = headerWrite(save_hdu, ds, None)
#  #        save_hdu.header['FILETYPE'] = 'KLIP mean cube'
#  #        save_hdu.header['DATASET'] = ds
#  #        save_hdu.header['INPFILE'] = hdr['FILENAME']
#  #        save_hdu.header['FILENAME'] = filename
#  #        save_hdu.header['NIMGAVG'] = (dataset.input.shape[0], 'number of images included in mean')
#  #        save_hdu.header['NUMBASIS'] = (str(numbasis), 'number of KL modes for each mean')
#  #        save_hdu.header['MAXNBASI'] = (maxnumbasis, 'max number of images to include for each KL mode')
#  #        save_hdu.header['ANNULI'] = (ann, 'number of annuli between IWA and OWA')
#  #        save_hdu.header['SUBSECTS'] = (subs, 'number of azimuthal subsections in each annulus')
#  #        save_hdu.header['MINROT'] = (minrot, 'minimum rotation threshold for reference images [deg]')
#  #        save_hdu.header['MOVEMENT'] = (mvmt, 'minimum movement threshold for reference images [pix]')
#  #        save_hdu.header['IWA'] = (IWA, 'inner edge of first KLIP annulus [pix]')
#  #        save_hdu.header['OWA'] = (OWA, 'outer edge of last KLIP annulus [pix]')
#  #        save_hdu.header['HIGHPASS'] = (highpass, 'size of high-pass filter box or sigma [pix]')
#  #        save_hdu.header['SPWIDTH'] = (spwidth, 'Pre PSF-sub diff. spike mask width (pix)')
#  #        save_hdu.header['PSPWIDTH'] = (ps_spwidth, 'Post PSF-sub diff. spike mask width (pix)')
#  #        save_hdu.header['STARPOSY'] = (dataset.aligned_center[1], 'Star y coordinate (numpy)')
#  #        save_hdu.header['STARPOSX'] = (dataset.aligned_center[0], 'Star x coordinate (numpy)')
#  #        save_hdu.header['PSFCENTY'] = (dataset.aligned_center[1], 'Star y coordinate (numpy)')
#  #        save_hdu.header['PSFCENTX'] = (dataset.aligned_center[0], 'Star x coordinate (numpy)')
#  #        save_hdu.header.add_history('= Created by T. Esposito, UCLA')
#  #        save_hdu.writeto(save)
#  #        print "KLIP mean cube saved as", save
#  #    except IOError:
#  #        print "WARNING: File already exists at", save,\
#  #                ", KLIP mean cube NOT SAVED to disk."
#  #    
#  #    if save_psf_cubes:
#  #        for ii, num_kl in enumerate(sub_imgs):
#  #            try:
#  #                if minrot is None:
#  #                    filename = 'a%ds%dmv%d_hp%s_KL%d_psfcube%s.fits' % (ann, subs, mvmt, str(highpass), numbasis[ii], sufx)
#  #                else:
#  #                    filename = 'a%ds%dmr%d_hp%s_KL%d_psfcube%s.fits' % (ann, subs, minrot, str(highpass), numbasis[ii], sufx)
#  #                save = os.path.expanduser('~/Research/data/%s/processed/%s' % (ds, filename))
#  #                os.path.exists(save)
#  #                save_hdu = fits.PrimaryHDU(data=num_kl.astype('float32'))
#  #                save_hdu = headerWrite(save_hdu, ds, None)
#  #                save_hdu.header['FILETYPE'] = 'KLIP PSF-subtracted cube'
#  #                save_hdu.header['DATASET'] = ds
#  #                save_hdu.header['INPFILE'] = hdr['FILENAME']
#  #                save_hdu.header['FILENAME'] = filename
#  #                save_hdu.header['NIMGAVG'] = (dataset.input.shape[0], 'number of images included in mean')
#  #                save_hdu.header['NUMBASIS'] = (numbasis[ii], 'number of KL modes used')
#  #                save_hdu.header['MAXNBASI'] = (maxnumbasis, 'max number of images to include for each KL mode')
#  #                save_hdu.header['ANNULI'] = (ann, 'number of annuli between IWA and OWA')
#  #                save_hdu.header['SUBSECTS'] = (subs, 'number of azimuthal subsections in each annulus')
#  #                save_hdu.header['MINROT'] = (minrot, 'minimum rotation threshold for reference images [deg]')
#  #                save_hdu.header['MOVEMENT'] = (mvmt, 'minimum movement threshold for reference images [pix]')
#  #                save_hdu.header['IWA'] = (IWA, 'inner edge of first KLIP annulus [pix]')
#  #                save_hdu.header['OWA'] = (OWA, 'outer edge of last KLIP annulus [pix]')
#  #                save_hdu.header['HIGHPASS'] = (highpass, 'size of high-pass filter box or sigma [pix]')
#  #                save_hdu.header['SPWIDTH'] = (spwidth, 'Pre PSF-sub diff. spike mask width (pix)')
#  #                save_hdu.header['PSPWIDTH'] = (ps_spwidth, 'Post PSF-sub diff. spike mask width (pix)')
#  #                save_hdu.header['STARPOSY'] = (dataset.aligned_center[1], 'Star y coordinate (numpy)')
#  #                save_hdu.header['STARPOSX'] = (dataset.aligned_center[0], 'Star x coordinate (numpy)')
#  #                save_hdu.header['PSFCENTY'] = (dataset.aligned_center[1], 'Star y coordinate (numpy)')
#  #                save_hdu.header['PSFCENTX'] = (dataset.aligned_center[0], 'Star x coordinate (numpy)')
#  #                save_hdu.header.add_history('= Created by T. Esposito, UCLA')
#  #                save_hdu.writeto(save)
#  #                print "KLIP PSF-subtracted cube saved as", save
#  #            except IOError:
#  #                print "WARNING: File already exists at", save,\
#  #                        ", KLIP PSF-subtracted cube NOT SAVED to disk."

#     if output:
#         return mean_arr
#     else:
#         pass
