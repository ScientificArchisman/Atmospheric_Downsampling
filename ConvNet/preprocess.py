import xarray as xr 
from skimage.transform import resize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd



low_res_path = "/Volumes/Extreme SSD/PRL/data/global4d/Jan2011/levtype_ml.nc"
high_res_path = "/Volumes/Extreme SSD/PRL/data/high_res/WRF_2011.nc"


def convert_time_stamps(time_stamps):
    return pd.to_datetime([t.decode('utf-8') if isinstance(t, bytes) 
                           else t for t in time_stamps], format='%Y-%m-%d_%H:%M:%S')

def resample_to_high_res(low_res_data, low_res_time, high_res_time, high_res_shape):
    """resample each low resolton data for every time stamp to match
    the dimensions of the high resolution data"""   
    resampled_data = []
    for img in low_res_data:
        resampled_data.append(resize(img, high_res_shape))
    resampled_data = np.array(resampled_data)

    # Convert datetime to float timestamps for interpolation
    high_res_time = high_res_time.astype(np.int64) // 10**9  # Convert to seconds since epoch
    low_res_time = low_res_time.astype(np.int64) // 10**9  # Convert to seconds since epoch

    interp_func = interp1d(low_res_time, resampled_data, axis = 0, kind='linear', fill_value='extrapolate')
    interpolated_ozone = interp_func(high_res_time)
    return interpolated_ozone


def crop_low_res_data(high_res_data, low_res_data):
    """ Crop the low-res data to match the high-res data
    :param high_res_data: xarray dataset
    :param low_res_data: xarray dataset
    :return: cropped_low_res_ozone: numpy array, low_res_lat: numpy array, low_res_long: numpy array,
             high_res_lat: numpy array, high_res_long: numpy array"""
    high_res_long, high_res_lat = high_res_data["XLONG"].values, high_res_data["XLAT"].values
    low_res_long, low_res_lat = low_res_data["longitude"].values, low_res_data["latitude"].values
    low_res_ozone = low_res_data["go3"].values

    # Find the indices of the low-res data that correspond to the high-res data
    low_res_long_indices = np.where((low_res_long >= high_res_long.min()) & (low_res_long <= high_res_long.max()))[0]
    low_res_lat_indices = np.where((low_res_lat >= high_res_lat.min()) & (low_res_lat <= high_res_lat.max()))[0]

    low_res_lat = low_res_lat[low_res_lat_indices]
    low_res_long = low_res_long[low_res_long_indices]

    # Crop the low-res data
    cropped_low_res_ozone = low_res_ozone[:, low_res_lat_indices.min() : low_res_lat_indices.max() + 1, low_res_long_indices.min() : low_res_long_indices.max() + 1]

    return cropped_low_res_ozone, low_res_lat, low_res_long, high_res_lat, high_res_long



def preprocess_data(low_res_path: str, high_res_path: str, 
                    low_res_pressure_level: int = 60, high_res_pressure_level: int = 0):
    """ Preprocess the data by normalizing and resampling the low resolution data
    to match the dimensions of the high resolution data.
    Args:
        low_res_path: str, path to the low resolution data
        high_res_path: str, path to the high resolution data
        low_res_pressure_level: int, pressure level for low resolution data. Default is 60
        high_res_pressure_level: int, pressure level for high resolution data. Default is 0
    Returns:
        low_res_ozone_resampled: np.array, resampled low resolution data
        high_res_ozone: np.array, high resolution data"""
    low_res_data = xr.open_dataset(low_res_path)
    high_res_data = xr.open_dataset(high_res_path)

    low_res_data = low_res_data.sel(level = low_res_pressure_level)
    high_res_data = high_res_data.sel(bottom_top = high_res_pressure_level)

    high_res_ozone = high_res_data["o3"].values
    low_res_ozone = low_res_data["go3"].values

    low_res_ozone, low_res_lat, low_res_long, high_res_lat, high_res_long = crop_low_res_data(high_res_data, low_res_data)

    high_res_time = high_res_data["Times"].values  # Convert high-res time data
    low_res_time = low_res_data["time"].values  # Convert low-res time data
    high_res_time = convert_time_stamps(high_res_time)
    low_res_time = convert_time_stamps(low_res_time)


    # Normalize the data
    high_res_ozone = (high_res_ozone - high_res_ozone.min()) / (high_res_ozone.max() - high_res_ozone.min())
    low_res_ozone = (low_res_ozone - low_res_ozone.min()) / (low_res_ozone.max() - low_res_ozone.min())

    high_res_shape = high_res_ozone.shape[1:]
    low_res_ozone_resampled = resample_to_high_res(low_res_ozone, 
                                                   low_res_time, 
                                                   high_res_time, 
                                                   high_res_shape)
    return low_res_ozone_resampled, high_res_ozone