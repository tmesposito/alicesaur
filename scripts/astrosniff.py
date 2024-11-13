from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from photutils.segmentation import detect_sources
import numpy as np
import pandas as pd
import os

def load_fits_data(self):
    """
    Loads the data from the final FITS image in the specified directory.
    """
    for filename in os.listdir(self.dataDir):
        if filename.startswith('final_') and filename.endswith('.fits'):
            file_path = os.path.join(self.dataDir, filename)
            with fits.open(file_path) as file:
                data = file[0].data[0]  # This varies depending on slice.
            return data, filename
    raise FileNotFoundError(f"No final FITS image found in the directory: {self.dataDir}")

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
    First step of the data processing using using convolve with gaussian kernel
    """
    kernel = make_2dgaussian_kernel(3, size=7)
    convolved_data = convolve(final_data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=10)
    seg_map.data[seg_map.data > 0] = 1
    return seg_map

def exclude_center_from_segmentation(seg_map, data, exclude_center=(1024, 1024), 
                                     exclude_radius=120):
    """
    Exclude a certain circular region from the center to avoid masking possible
    disks.
    """
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    distance_from_center = np.sqrt((x - exclude_center[0])**2 + (y - exclude_center[1])**2)
    exclusion_mask = distance_from_center <= exclude_radius
    seg_map.data[exclusion_mask] = 0
    return seg_map

def masked_pixels_coords(seg_map, csv_file_path='masked_pixel_yx.csv'):

    all_pixels = np.argwhere(seg_map.data == 1)
    df = pd.DataFrame(all_pixels, columns=['y', 'x'])
    df.to_csv(csv_file_path, index=False)
    print(f"Saved masked pixel coordinates to {csv_file_path}")

def save_seg_map_as_fits(seg_map, original_filename, output_directory='.'):

    output_file = f"segmap_{original_filename.split('final_')[1]}"
    output_path = os.path.join(output_directory, output_file)
    hdu = fits.PrimaryHDU(seg_map.data)
    hdu.writeto(output_path, overwrite=True)
    print(f"Segmentation map saved as {output_path}")

def main_masking(self):

    data, original_filename = load_fits_data(self)
    final_data, threshold = subtract_background(data)
    seg_map = process_data(final_data, threshold)
    seg_map = exclude_center_from_segmentation(seg_map, data)
    csv_path = os.path.join(self.dataDir, 'masked_pixel_yx.csv')
    masked_pixels_coords(seg_map, csv_path)
    save_seg_map_as_fits(seg_map, original_filename, output_directory=self.dataDir)

    return seg_map
