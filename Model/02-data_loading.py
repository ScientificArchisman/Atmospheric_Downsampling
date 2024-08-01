import torch 
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np 
import xarray as xr


class WRFDataset(Dataset):
    def __init__(self, high_res_data, low_res_data, chunk_size_lat, chunk_size_long):
        self.high_res_data = high_res_data
        self.low_res_data = low_res_data
        self.chunk_size_lat = chunk_size_lat
        self.chunk_size_long = chunk_size_long
        
        # Ensure both datasets have the same shape
        assert high_res_data.shape == low_res_data.shape, "High-res and low-res data must have the same shape"
        
        # Calculate the number of chunks
        self.n_chunks_lat = high_res_data.shape[2] // chunk_size_lat
        self.n_chunks_long = high_res_data.shape[3] // chunk_size_long

        # Calculate the total number of chunks
        self.n_chunks = self.n_chunks_lat * self.n_chunks_long

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # Calculate the chunk's starting indices for latitude and longitude
        lat_idx = idx // self.n_chunks_long
        long_idx = idx % self.n_chunks_long
        
        lat_start = lat_idx * self.chunk_size_lat
        lat_end = lat_start + self.chunk_size_lat
        long_start = long_idx * self.chunk_size_long
        long_end = long_start + self.chunk_size_long
        
        high_res_chunk = self.high_res_data[:, :, lat_start:lat_end, long_start:long_end]
        low_res_chunk = self.low_res_data[:, :, lat_start:lat_end, long_start:long_end]
        
        return torch.tensor(high_res_chunk, dtype=torch.float32), torch.tensor(low_res_chunk, dtype=torch.float32)


def create_loaders(dataset, batch_size: int = 16):
    # Split indices
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, total_size - train_size])

    valid_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # ozone_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_2011.nc")
    # co_no2_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_CO_NO2.nc")
    # no_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_NO.nc")
    # humidity_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_SpecificHum.nc")
    # temp_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_2011_Archi_T.nc")
    # TIME_POINTS, PRESSURE_POINTS, LAT_POINTS, LON_POINTS = ozone_2011["o3"].shape

    # combined_data = xr.Dataset(
    #     {
    #         "ozone": ozone_2011["o3"],
    #         "pm25": ozone_2011["PM2_5_DRY"],
    #         "co":  co_no2_2011["co"],
    #         "no2": co_no2_2011["no2"],
    #         "no": no_2011["no"],
    #         "humidity": humidity_2011["QVAPOR"],
    #         "temperature": temp_2011["T2"].expand_dims({"bottom_top": np.arange(PRESSURE_POINTS)}, axis=1)
    #     },
    #     coords={"time": ozone_2011["Times"], 
    #             "pressure": ozone_2011["bottom_top"], 
    #             "latitude": ozone_2011["south_north"], 
    #             "longitude": ozone_2011["west_east"]})

    ## DATASHAPE: (N_VARIABLES, TIME_POINTS, LATITUDE_POINTS, LONGITUDE_POINTS)
    high_res_data = np.random.rand(2, 100, 1024, 1024)
    low_res_data = np.random.rand(2, 100, 1024, 1024)

    dataset = WRFDataset(high_res_data, low_res_data, 16, 16)
    train_loader, valid_loader, test_loader = create_loaders(dataset, 16)

    for i, (high_res_chunk, low_res_chunk) in enumerate(train_loader):
        print(f"Batch {i}")
        print(f"High-res chunk shape: {high_res_chunk.shape}")
        print(f"Low-res chunk shape: {low_res_chunk.shape}")
        break

    
