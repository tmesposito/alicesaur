import os
import numpy as np
from astropy.modeling import models, fitting

def tie_std(model):
    return model.x_stddev


def fit(im, x, y, source_id, exclude_id, star_errors, offset, inflation,
        method='gaussian', data=None):
    """

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    source_id : TYPE
        DESCRIPTION.
    exclude_id : TYPE
        DESCRIPTION.
    star_errors : TYPE
        DESCRIPTION.
    offset : list of float
        [X, Y] offsets in [pixels] to be applied to all fit center outputs
    inflation : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'gaussian'.
    data : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    list
        DESCRIPTION.
    model_stamps : TYPE
        DESCRIPTION.
    model_fits : TYPE
        DESCRIPTION.

    """

    # We assume that the input list has been cleaned and these are all valid stars to fit

    n = len(x)
    px_pos = []
    px_cov = []
    data1_stamps = []
    data2_stamps = []
    model_stamps = []
    model_fits = []

    xoff, yoff = offset
    xinf, yinf = inflation

    if method == 'gaussian':

        pad1 = 5
        pad2 = 5

        if star_errors is None:
            print("Assuming 0.1 pixel uncertainty on all Gaia star "\
                  "centers because no measured uncertainties were provided")

        for i in range(0, n):
            if source_id[i] in exclude_id:
                px_pos.append(np.full(2, np.nan))
                px_cov.append(np.full((2, 2), np.nan))
                data1_stamps.append(None)
                data2_stamps.append(None)
                model_stamps.append(None)
                model_fits.append(None)
            else:

                # First round
                x1 = int(x[i])
                y1 = int(y[i])
                stamp1 = im[y1-pad1:y1+pad1+1, x1-pad1:x1+pad1+1]
                weight1 = np.full(stamp1.shape, 1.0)
                weight1[np.where(np.isnan(stamp1))] = 0.0
                stamp1[np.where(np.isnan(stamp1))] = 0.0

                gauss_model = models.Gaussian2D(amplitude=np.nanmax(stamp1), x_mean=pad1, y_mean=pad1,
                                                x_stddev=2.0, y_stddev=2.0,
                                                tied = {'y_stddev': tie_std},
                                                fixed = {'theta': True})

                least_sq_fit = fitting.LevMarLSQFitter()
                yarr,xarr = np.indices(stamp1.shape)
                best_fit_model = least_sq_fit(gauss_model, xarr, yarr, stamp1, weights=weight1)
               
                # Second round
                x2 = int(np.round(best_fit_model.x_mean)) + (x1-pad1)
                y2 = int(np.round(best_fit_model.y_mean)) + (y1-pad1)

                stamp2 = im[y2-pad2:y2+pad2+1, x2-pad2:x2+pad2+1]
                weight2 = np.full(stamp2.shape, 1.0)
                weight2[np.where(np.isnan(stamp2))] = 0.0
                stamp2[np.where(np.isnan(stamp2))] = 0.0

                gauss_model = models.Gaussian2D(amplitude=np.nanmax(stamp2), x_mean=pad2, y_mean=pad2,
                                                x_stddev=2.0, y_stddev=2.0,
                                                tied = {'y_stddev': tie_std},
                                                fixed = {'theta': True})

                least_sq_fit = fitting.LMLSQFitter(calc_uncertainties=False)
                yarr,xarr = np.indices(stamp2.shape)
                best_fit_model = least_sq_fit(gauss_model, xarr, yarr, stamp2, weights=weight2)

                px_pos.append((best_fit_model.x_mean + (x2-pad2) + xoff, best_fit_model.y_mean + (y2-pad2) + yoff))

                # Save the model and stamps
                model_fits.append(best_fit_model)
                data1_stamps.append(stamp1)
                data2_stamps.append(stamp2)
                model_stamps.append(best_fit_model(xarr, yarr))

                # Prefer to use measured uncertainties for star centers. If
                # none exist, assume 0.1 pixel error for all.
                if star_errors is not None:
                    err_indx = np.where((star_errors['gaia_id'].data == data['Source'][j]) & (star_errors['file'].data == os.path.basename(im_list[i])))[0][0]
                    x_err = star_errors['x_err'].data[err_indx]
                    y_err = star_errors['y_err'].data[err_indx]
                else:
                    x_err = 0.1
                    y_err = 0.1

                # xinf, yinf = error inflation terms
                if xinf > 0.0: x_err = np.sqrt((x_err**2 + xinf**2))
                if yinf > 0.0: y_err = np.sqrt((y_err**2 + yinf**2))

                px_cov.append(np.diag((x_err**2, y_err**2)))

    return np.array(px_pos), np.array(px_cov), [data1_stamps, data2_stamps], model_stamps, model_fits
