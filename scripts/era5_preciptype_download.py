import cdsapi
import os
from numpy import arange

start_year = int(os.environ['start_year'])
end_year_ex = int(os.environ['end_year_ex']) # exclusive of this year

print('Starting PRECIP_TYPE',flush = True)
year_list = list(arange(start_year,end_year_ex).astype(str))
for year_str in year_list:
    output_file = '/global/scratch/users/ann_scheliga/era5_data/' + year_str + 'daily_precip_type.nc'
    print('Starting', year_str,flush=True)
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": ["precipitation_type"],
        "year": year_str,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        "area": [40, -180, -40, 180]
    }
    
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target = output_file)
    print('Saved to', output_file,flush=True)
