#!/usr/bin/env python

import pdb
import numpy as np
from scipy.ndimage import median_filter, generic_filter

# Internal imports
from alicesaur import utils


def mask_bad_pix(im, inst=None, Nsig=7, neighborDist=5, thr_min=np.inf,
                 low_only=False, negAlwaysBad=False,
                 iterate=False, iterThresh=0.0005, window=False):
    """
    Remove bad pixels by comparing their value to mean of neighbor pixels.

    Inputs:
        im= image array to mark bad pixels in.
        inst= str; name of instrument im came from; only needed if instrument
                has known bad pixel locations.
        Nsig= # of standard deviations above the mean required to mark pixel as bad.
        thr_min= hard minimum value above the mean required to mark pixel as bad.
        low_only: bool, True to only find outliers below the mean of neighbors.
        negAlwaysBad: bool, True to always flag negative pixels as bad.
        iterate: bool, True to iterate until fraction of bad pixels removed
            is < iterThresh.
        iterThresh: float, fraction of bad pixels allowed to stay.

    Output:
        numpy array with bad pixels = median of nearest neighbors.
    """

    print("\nMarking bad pixels (may take a few minutes)...")

    # Create zero arrays same size as im but padded by 2 pixels on each side.
    # Shift im around edge of an 8-pixel square, with starting point
    # (0,0), to make 16 new arrays with differentially-shifted im.
    # for yy in [-3, -2, 2, 3]:
    #     for xx in [-3, -2, 2, 3]:

    if window:
        patch = im[window[0]-neighborDist:window[1]+neighborDist+1,
                   window[2]-neighborDist:window[3]+neighborDist+1]
    else:
        patch = im

    yx_list = []
    for yy in range(-neighborDist, neighborDist+1): #[-3, -2, -1, 0, 1, 2, 3]:
            for xx in range(-neighborDist, neighborDist+1): #[-3, -2, -1, 0, 1, 2, 3]:
                if not (yy==0 and xx==0):
                    yx_list.append((yy,xx))

    # plt.figure(0)
    # plt.clf()
    # plt.imshow(im, vmin=-100, vmax=100)
    # plt.title('Original')
    # plt.draw()

    badFrac = 1. # initialize fraction of bad pixels at 100%
    it = 0 # iteration count
    while badFrac > iterThresh:
        mat_list = []

        for yx in yx_list:
            if yx==(0,0):
                continue
            # mat = np.zeros((im.shape[0]+4, im.shape[1]+4))
            # mat[yx[0]:yx[0]+im.shape[0], yx[1]:yx[1]+im.shape[1]] = im.copy()
            mat = utils.shift_im_center(patch, (patch.shape[0]/2,patch.shape[1]/2),
                        (patch.shape[0]/2 + yx[0],patch.shape[1]/2 + yx[1]),
                        fillval=np.nan, size_out=None)
            mat_list.append(mat)

        mat_list = np.array(mat_list)

        # Calculate the mean of nearest neighbors (ignore NaN).
        # # Calculate standard deviation of nearest neighbors (ignore NaN).
        # # (Note: equivalent to deprecated scipy.stats.nanstd(mat_list, axis=0, bias=True))
        # nebStd_mat = nanstd(mat_list, axis=0)
        # nebMeanArr = nanmean(mat_list, axis=0)
        # nebMedianArr = np.nanmedian(mat_list, axis=0)
        filtSize = 5

        nebMedianArr = median_filter(patch, size=filtSize, mode='reflect')
        # Calculate standard deviation of nearest neighbors (ignore NaN).
        # (Note: equivalent to deprecated scipy.stats.nanstd(mat_list, axis=0, bias=True))
        nebStdArr = np.nanstd(mat_list, axis=0)

        # # Trim padding off of nebMean_mat and nebStd_mat so they match shape of im.
        # # NOTE: intentionally starting at index 1, based on shifting above.
        # nebMeanArr = nebMean_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # nebMedianArr = nebMedian_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # nebStdArr = nebStd_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # Set threshold requirement for a "bad" pixel as Nsig standard deviations
        # above or below the smaller of mean & thr_min.
        thresh = Nsig*nebStdArr.copy()
        thresh[thresh > thr_min] = thr_min
        # Find all pixels in im that exceed thresh relative to neighbors.
        # diff = np.abs(nebMeanArr - im) - thresh
        if low_only:
            diff = nebMedianArr - patch
            diff[diff < 0] = 0
            diff -= thresh
            whb = np.where(diff > 0)
        else:
            diff = np.abs(nebMedianArr - patch) - thresh
            if negAlwaysBad:
                whb = np.where((diff > 0)  | (patch < 0))
            else:
                whb = np.where((diff > 0)  | (patch < -5*np.nanmean(nebStdArr)))

        # # Replace all "bad" pixels with NaN.
        # im[whb] = np.nan
        # Replace all "bad" pixels with median of neighbors.
        patch[whb] = nebMedianArr[whb]

        if window:
            im[window[0]-neighborDist:window[1]+neighborDist+1, window[2]-neighborDist:window[3]+neighborDist+1] = patch

        Nbad = whb[0].shape[0]
        badFrac = whb[0].shape[0]/float(patch.size)
        # # Mask known bad pixels according to instrument specified.
        # if inst=="NickelDIC":
        #     im[:, (256, 783, 784)] = np.nan

        print("Iter %d: %d bad pixels (%.2f%%) fixed (not counting known bad pixels)" % (it, Nbad, 100*badFrac))

        if not iterate:
            badFrac = 0.
        else:
            it += 1

        # sigmaMap = np.abs((nebMedianArr - patch)/nebStdArr)
        # 
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(nebStdArr, vmin=-10, vmax=10)
        # plt.title('Std dev')
        # plt.draw()
        # 
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(nebMedianArr, vmin=-100, vmax=100)
        # plt.title('Median filtered')
        # plt.draw()
        # 
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(patch - nebMedianArr, vmin=-10, vmax=10)
        # plt.title('Original - Median')
        # plt.draw()
        # 
        # plt.figure(4)
        # plt.clf()
        # plt.imshow(sigmaMap, vmin=-Nsig, vmax=Nsig)
        # plt.title('Sigma map ({} to {})'.format(-Nsig, Nsig))
        # plt.draw()
        # 
        # plt.figure(5)
        # plt.clf()
        # plt.imshow(patch, vmin=-100, vmax=100)
        # plt.title('Fixed original')
        # plt.draw()
        # 
        # pdb.set_trace()

    return im


def interp_bad_pix(im, inst=None, Nsig=5, filtSize=5, thr_min=np.inf, low_only=False,
                 iterate=False, iterThresh=0.0005):
    """
    Remove bad pixels by interpolating over them.

    Inputs:
        im= image array to mark bad pixels in.
        inst= str; name of instrument im came from; only needed if instrument
                has known bad pixel locations.
        Nsig= # of standard deviations above the mean required to mark pixel as bad.
        thr_min= hard minimum value above the mean required to mark pixel as bad.
        low_only: bool, True to only find outliers below the mean of neighbors.
        iterate: bool, True to iterate until fraction of bad pixels removed
            is < iterThresh.
        iterThresh: float, fraction of bad pixels allowed to stay.

    Output:
        numpy array with bad pixels = NaN.
    """

    print("\nMarking bad pixels (may take a few minutes)...")

    # Create zero arrays same size as im but padded by 2 pixels on each side.
    # Shift im around edge of an 8-pixel square, with starting point
    # (0,0), to make 16 new arrays with differentially-shifted im.
    # for yy in [-3, -2, 2, 3]:
    #     for xx in [-3, -2, 2, 3]:

    yx_list = []
    # for yy in [-2, -1, 0, 1, 2]:
    #         for xx in [-2, -1, 0, 1, 2]:
    #             if not (yy==0 and xx==0):
    #                 yx_list.append((yy,xx))

    badFrac = 1. # initialize fraction of bad pixels at 100%
    it = 0 # iteration count
    while badFrac > iterThresh:
        # mat_list = []
        # 
        # for yx in yx_list:
        #     if yx==(0,0):
        #         continue
        #     # mat = np.zeros((im.shape[0]+4, im.shape[1]+4))
        #     # mat[yx[0]:yx[0]+im.shape[0], yx[1]:yx[1]+im.shape[1]] = im.copy()
        #     mat = utils.shift_im_center(im, (im.shape[0]/2,im.shape[1]/2),
        #                 (im.shape[0]/2 + yx[0],im.shape[1]/2 + yx[1]),
        #                 fillval=np.nan, size_out=None)
        #     mat_list.append(mat)
        # 
        # mat_list = np.array(mat_list)
        # 
        # # Calculate the mean of nearest neighbors (ignore NaN).
        # # # Calculate standard deviation of nearest neighbors (ignore NaN).
        # # # (Note: equivalent to deprecated scipy.stats.nanstd(mat_list, axis=0, bias=True))
        # # nebStd_mat = nanstd(mat_list, axis=0)
        # # nebMeanArr = nanmean(mat_list, axis=0)
        # nebMedianArr = np.nanmedian(mat_list, axis=0)
        # # Calculate standard deviation of nearest neighbors (ignore NaN).
        # # (Note: equivalent to deprecated scipy.stats.nanstd(mat_list, axis=0, bias=True))
        # nebStdArr = np.nanstd(mat_list, axis=0)

        # This is <1 s per iteration.
        nebMedianArr = median_filter(im, size=filtSize, mode='reflect')

        def stddev(x):
            return np.sqrt(np.sum((x - np.mean(x))**2)/np.size(x))

        breakpoint()
        test = generic_filter(im, stddev, size=filtSize)
        breakpoint()
        # This is absurdly slow... ~30 seconds per iteration.
        nebStdArr = generic_filter(im, np.std, size=filtSize)
        breakpoint()

        # plt.figure(0)
        # plt.clf()
        # plt.imshow(im, vmin=-100, vmax=100)
        # plt.title('Original')
        # plt.draw()

        # # # Trim padding off of nebMean_mat and nebStd_mat so they match shape of im.
        # # # NOTE: intentionally starting at index 1, based on shifting above.
        # # nebMeanArr = nebMean_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # # nebMedianArr = nebMedian_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # # nebStdArr = nebStd_mat[1:im.shape[0]+1, 1:im.shape[1]+1]
        # # Set threshold requirement for a "bad" pixel as Nsig standard deviations
        # # above or below the smaller of mean & thr_min.
        # thresh = Nsig*nebStdArr.copy()
        # thresh[thresh > thr_min] = thr_min
        # Find all pixels in im that exceed thresh relative to neighbors.
        # diff = np.abs(nebMeanArr - im) - thresh
        if low_only:
            diff = nebMedianArr - im
            diff[diff < 0] = 0
            diff -= thresh
            whb = np.where(diff > 0)
        else:
            # diff = np.abs(nebMedianArr - im) - thresh
            # whb = np.where(diff > 0)
            diff = im - nebMedianArr
            whb = np.where(((np.abs(diff)/nebStdArr) > Nsig) | (im < -5*np.nanmean(nebStdArr)))

        # # # Replace all "bad" pixels with NaN.
        # # im[whb] = np.nan
        # # Replace all "bad" pixels with median of neighbors.
        im[whb] = nebMedianArr[whb]

        Nbad = whb[0].shape[0]
        badFrac = whb[0].shape[0]/float(im.size)
        # # Mask known bad pixels according to instrument specified.
        # if inst=="NickelDIC":
        #     im[:, (256, 783, 784)] = np.nan

        print("Iter %d: %d bad pixels (%.1f%%) marked as NaN (not counting known bad pixels)" % (it, Nbad, 100*badFrac))

        if not iterate:
            badFrac = 0.
        else:
            it += 1

        # sigmaMap = np.abs((nebMedianArr - im)/nebStdArr)
        # 
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(nebStdArr, vmin=-10, vmax=10)
        # plt.title('Std dev')
        # plt.draw()
        # 
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(nebMedianArr, vmin=-100, vmax=100)
        # plt.title('Median filtered')
        # plt.draw()
        # 
        # plt.figure(3)
        # plt.clf()
        # plt.imshow(im - nebMedianArr, vmin=-10, vmax=10)
        # plt.title('Original - Median')
        # plt.draw()
        # 
        # plt.figure(4)
        # plt.clf()
        # plt.imshow(sigmaMap, vmin=-Nsig, vmax=Nsig)
        # plt.title('Sigma map ({} to {})'.format(-Nsig, Nsig))
        # plt.draw()
        # 
        # plt.figure(5)
        # plt.clf()
        # plt.imshow(im, vmin=-100, vmax=100)
        # plt.title('Fixed original')
        # plt.draw()
        # 
        # breakpoint()

    return im


def fix_bad_pix(imgs, intensify=False):
    """
    Correct bad pixels.
    
    intensify: Focus intense bad pixel fixing on a specific rectangle in the images
        in a second round of iterations. To do so, give this argument a list of
        [Y min, x min, box height, box width] all in pixels.
        Default of False does not do this intense step.
    """

    imgsBadMasked = [mask_bad_pix(imgs[ii], inst=None, Nsig=5,
                                  neighborDist=4, low_only=False,
                                  iterate=True)
                     for ii in range(len(imgs))]

    if intensify:
        neighborDist = 2
        for ii in range(len(intensify)):
            if intensify[ii] < neighborDist:
                intensify[ii] = neighborDist
        ymin, xmin, he, wi = intensify
        print("\nRunning 'intense' bad pixel fixing...")
        imgsBadMasked = [mask_bad_pix(imgsBadMasked[ii], inst=None, Nsig=5,
                                      neighborDist=neighborDist, low_only=False,
                                      negAlwaysBad=True,
                                      iterate=True, iterThresh=0.,
                                      window=[ymin, ymin+he, xmin, xmin+wi])
                         for ii in range(len(imgs))]

    # plt.figure(20)
    # plt.clf()
    # plt.imshow(imgsBadMasked[0], norm=SymLogNorm(linthresh=1, linscale=1,
    #                                              vmin=0, vmax=100000))
    # plt.ylim(ymin, ymin+he)
    # plt.xlim(xmin, xmin+wi)
    # plt.draw()
    # 
    # pdb.set_trace()

    return np.array(imgsBadMasked)
