import cdsapi
from numpy import arange

print('Starting PRECIP TYPE',flush=True)

year_list = list(arange(2019,2020).astype(str))
for year_str in year_list:
    output_file = '/global/scratch/users/ann_scheliga/era5_test_data/' + year_str + 'daiy_precip_type.nc'
    print('Starting', year_str,flush=True)
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
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
        "daily_statistic": "daily_sum",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly",
        "area": [40, -180, -40, 180]
    }
    
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target = output_file)
    print('Saved to', output_file,flush=True)
