from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from photutils.segmentation import detect_sources
import numpy as np
import pandas as pd
import os

def load_fits_data():
    for filename in os.listdir('.'):
        if filename.startswith('final') and filename.endswith('.fits'):
            with fits.open(filename) as file:
                data = file[0].data[0]
            return data #!!!data[i] acceses the (i)th slice of the fits image!!!
    raise FileNotFoundError("There is no final image in the directory.")

def subtract_background(data):
    # Set fixed sigma
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    
    # Estimate the background value
    bkg = Background2D(data, (250, 250), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    
    # Subtract background from the data
    final_data = data - bkg.background
    
    # Calculate the threshold
    threshold = 3 * np.median(bkg.background_rms)
    
    return final_data, threshold

def process_data(final_data, threshold):
    # Create a 2D Gaussian kernel and convolve the data
    kernel = make_2dgaussian_kernel(3, size=7)
    convolved_data = convolve(final_data, kernel)
    
    # Detect sources using the provided threshold and apply masking
    seg_map = detect_sources(convolved_data, threshold, npixels=10)
    seg_map.data[seg_map.data > 0] = 1
    
    return seg_map

def exclude_center_from_segmentation(seg_map, data, exclude_radius=50):
    """
    Exclude pixels within a specified radius from a fixed center point in the segmentation map.
    
    Parameters:
        seg_map: The segmentation map to modify.
        data: The original data to determine the shape.
        exclude_radius: The radius around the center to exclude.

    Returns:
        Modified segmentation map with excluded areas set to 0.
    """
    # Fixed center coordinates to exclude
    exclude_center = (1024, 1024)

    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    distance_from_center = np.sqrt((x - exclude_center[0])**2 + (y - exclude_center[1])**2)
    exclusion_mask = distance_from_center <= exclude_radius
    seg_map.data[exclusion_mask] = 0
    
    return seg_map

def masked_pixels_coords(seg_map, csv_file_path='masked_pixel_yx.csv'):
    """
    Save the pixel coords from the segmentation map to a CSV file.

    Parameters:
        seg_map: The segmentation map from which to extract pixel coords.
        csv_file_path: The path where the CSV file will be saved.

    """
    all_pixels = np.argwhere(seg_map.data == 1)

    df = pd.DataFrame(all_pixels, columns=['y', 'x'])  # Note: Using (y, x) as per image coordinate convention
    df.to_csv(csv_file_path, index=False)

    print(f"Saved masked pixel coordinates to {csv_file_path}")


def save_seg_map_as_fits(seg_map, output_file='seg_map.fits'):
    """
    Save the segmentation map as a .fits file.
    
    Parameters:
        seg_map: The 2D array with the masking done (the segmentation map)
    """
    
    hdu = fits.PrimaryHDU(seg_map.data)
    hdu.writeto(output_file, overwrite=True)
    print(f"Segmentation map saved as {output_file}")

def main():
    data = load_fits_data()
    final_data, threshold = subtract_background(data)
    seg_map = process_data(final_data, threshold)
    seg_map = exclude_center_from_segmentation(seg_map, data)
    masked_pixels_coords(seg_map)
    save_seg_map_as_fits(seg_map)
    
    return seg_map

if __name__ == "__main__":
    seg_map = main()