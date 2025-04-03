from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from photutils.segmentation import detect_sources
import numpy as np
import pandas as pd
import os


def load_fits_data(dataDir):
    """
    Loads the data from the final FITS image in the specified directory.
    """
    for filename in sorted(os.listdir(dataDir)):
        if filename.startswith('final_') and filename.endswith('.fits'):
            try:
                file_path = os.path.join(dataDir, filename)
                with fits.open(file_path) as hdul:
                    data_cube = hdul[1].data
                    # If there are multiple data slices, use the first.
                    if data_cube.ndim == 3:
                        data = data_cube[0]
                    else:
                        data = data_cube
                    hdr = hdul[0].header
                    star = np.array([hdr['PSFCENTY'], hdr['PSFCENTX']])

                return data, filename, star
            except:
                continue

    return None, None, None

def subtract_background(data):
    """
    The process for the background substraction and threshold calculation.
    """
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (250, 250), filter_size=(3, 3), sigma_clip=sigma_clip, 
                       bkg_estimator=bkg_estimator)
    final_data = data - bkg.background
    threshold = 3 * np.median(bkg.background_rms)
    return final_data, threshold

def process_data(final_data, threshold):
    """
    First step of the data processing using convolution with gaussian kernel.
    """
    kernel = make_2dgaussian_kernel(3, size=7)
    convolved_data = convolve(final_data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=10)
    seg_map.data[seg_map.data > 0] = 1
    return seg_map

def exclude_center_from_segmentation(seg_map, data,
                                     exclude_center=(1024, 1024),
                                     exclude_radius=120):
    """
    Exclude a certain circular region from the center to avoid masking possible
    disks.

    Inputs:

        exclude_radius: scalar
          Radius from the exclude_center to avoid masking, in [pixels].

    Outputs:

        A 2-d array "segmentation map" with value 1 for all masked pixels and
        0 for all unmasked pixels.

    """
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    distance_from_center = np.sqrt((x - exclude_center[0])**2 + (y - exclude_center[1])**2)
    exclusion_mask = distance_from_center <= exclude_radius
    seg_map.data[exclusion_mask] = 0

    return seg_map

def masked_pixels_coords(seg_map, csv_file_path='masked_pixel_yx.csv'):

    all_pixels = np.argwhere(seg_map.data == 1)
    df = pd.DataFrame(all_pixels, columns=['y', 'x'])

    if csv_file_path is not None:
        df.to_csv(csv_file_path, index=False)
        print(f"Saved masked pixel coordinates to {csv_file_path}")

    return df

def save_seg_map_as_fits(seg_map, input_filename, output_directory='.'):

    output_file = f"segmap_{os.path.splitext(input_filename)[0]}.fits"
    output_path = os.path.join(output_directory, output_file)
    hdu = fits.PrimaryHDU(seg_map.data)
    hdu.writeto(output_path, overwrite=True)
    print(f"Segmentation map saved as {output_path}")
    return

def main_masking(data=None, dataDir='.', input_filename='image',
                 star=None, exclude_radius=120):

    if data is None:
        data, input_filename, star = load_fits_data(dataDir=dataDir)

    if data is not None:
        print(f"Auto star masking: Using input image {input_filename}")
        final_data, threshold = subtract_background(data)
        seg_map = process_data(final_data, threshold)
        if star is not None:
            seg_map = exclude_center_from_segmentation(seg_map, data,
                                                 exclude_center=star,
                                                 exclude_radius=exclude_radius)
        # Save the coordinates of the masked pixels to a csv table.
        csv_path = os.path.join(dataDir, f'masked_pixel_yx_{os.path.splitext(input_filename)[0]}.csv')
        masked_pixels_coords(seg_map, csv_path)
        # Save the mask to a FITS image.
        save_seg_map_as_fits(seg_map, input_filename,
                             output_directory=dataDir)
        return seg_map
    else:
        return None
