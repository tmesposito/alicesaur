from matplotlib import pyplot as plt
from matplotlib import colors, cm
import numpy as np
from alicesaur.gaia import gaia_utils

def plot_overview(im, x, y, source_id, exclude_id, outname='gaia_overview'):

    fig, ax = plt.subplots(1, figsize=(5, 5))

    vmin, vmax = np.nanpercentile(im[np.where(im != 0.0)], [5, 99])
    ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax)

    s = im.shape
    xlim = (0, s[1])
    ylim = (0, s[0])

    # Loop through each Gaia source and plot
    for j in range(0, len(x)):
        if source_id[j] in exclude_id:
            marker, mec = 'x', 'red'
        else:
            marker, mec = 'o', 'green'

        ax.plot(x[j], y[j], marker, mec=mec, ms=5, mew=0.5, mfc='None')
        ax.annotate(str(j), xy=(x[j], y[j]), xycoords='data', color=mec, xytext=(4, 4), textcoords='offset points', fontsize=8)
    
    fig.savefig('{}.png'.format(outname), dpi=300, bbox_inches='tight') # TODO: Change to meaningful name
    plt.close('all')

    return 0

def plot_fits(data_table, include_indx, px_pos, px_cov, data_stamps, model_stamps, model_fits, samples, lnp, blobs, xoff, yoff, outname='gaia_psffits'):

    n_stars = len(include_indx)
    fig, ax = plt.subplots(n_stars, 6, figsize=(12, 2.0*float(n_stars)))

    n_steps, n_walkers, n_dim = samples.shape

    min_ind = np.unravel_index(np.nanargmax(lnp), lnp.shape)
    best_p = samples[min_ind[0], min_ind[1]]

    for j, i in enumerate(include_indx):
        vmin, vmax = np.nanmin(data_stamps[1][i]), np.nanmax(data_stamps[1][i])

        _ = ax[j][0].imshow(data_stamps[0][i], origin='lower', vmin=vmin, vmax=vmax)
        _ = ax[j][1].imshow(data_stamps[1][i], origin='lower', vmin=vmin, vmax=vmax)
        _ = ax[j][1].plot(model_fits[i].x_mean + xoff, model_fits[i].y_mean + yoff, 'x', ms=5, mew=0.5, color='white')
        _ = ax[j][2].imshow(model_stamps[i], origin='lower', vmin=vmin, vmax=vmax)
        _ = ax[j][3].imshow(data_stamps[1][i] - model_stamps[i], origin='lower', vmin=vmin, vmax=vmax)

        n_draws = int(1e3)
        n_use = 25
        pred_pos = np.zeros((n_use, n_walkers, n_draws, 2))

        for i1 in range(n_steps-n_use, n_steps):
            for i2 in range(0, n_walkers):
                pred_pos[i1-(n_steps-n_use), i2] = np.random.multivariate_normal(blobs[1][i1][i2][j], blobs[2][i1][i2][j], n_draws)

        pred_x = pred_pos[...,0].flatten()
        pred_y = pred_pos[...,1].flatten()
        pred_pos = None

        std_x, std_y = np.std(pred_x), np.std(pred_y)
        std = np.max((std_x, std_y)) * 8.0
        xbins = np.linspace(np.mean(pred_x)-std, np.mean(pred_x)+std, 100)
        ybins = np.linspace(np.mean(pred_y)-std, np.mean(pred_y)+std, 100)

        h, _, _, _ = ax[j][4].hist2d(pred_x, pred_y, cmap=cm.Greys, bins=(xbins, ybins))
        _ = ax[j][4].contour(gaia_utils.confidence_levels(h.T), [1.99, 2.99, 3.99], extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()])

        ## And plot pixel measurement
        _ = ax[j][4].errorbar(px_pos[i][0], px_pos[i][1], 
            xerr=np.sqrt(px_cov[i][0][0]),
            yerr=np.sqrt(px_cov[i][1][1]), fmt='o', color='red', ms=2.0, capsize=5)

        _ = ax[j][4].xaxis.get_major_formatter().set_useOffset(False)
        _ = ax[j][4].yaxis.get_major_formatter().set_useOffset(False)

        _ = ax[j][5].annotate('{}'.format(data_table['source_id'][i]), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=6)
        _ = ax[j][5].annotate(r'$\chi^2_i =$ {:.1f}'.format(blobs[0][min_ind[0]][min_ind[1]][j]), xy=(0.05, 0.75), xycoords='axes fraction')
        _ = ax[j][5].axis('off')

        color = 'k'
        _ = ax[j][5].annotate(r'$A =$ {:.1f}'.format(model_fits[i].amplitude.value), color=color, xy=(0.05, 0.60), xycoords='axes fraction')
        _ = ax[j][5].annotate(r'$\sigma =$ {:.2f}'.format(model_fits[i].x_stddev.value), xy=(0.05, 0.45), xycoords='axes fraction')
        _ = ax[j][5].annotate(r'FWHM $=$ {:.2f}'.format(model_fits[i].x_stddev.value*2.355), xy=(0.05, 0.30), xycoords='axes fraction')
        _ = ax[j][5].annotate(f'Stamp {i}', xy=(0.05, 0.15), xycoords='axes fraction')

    _ = ax[0][0].set_title('Data')
    _ = ax[0][1].set_title('Data (re-centered)')
    _ = ax[0][2].set_title('Model')
    _ = ax[0][3].set_title('Residual')

    fig.savefig('{}.png'.format(outname), dpi=300, bbox_inches='tight')

    return 0