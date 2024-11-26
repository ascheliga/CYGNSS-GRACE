import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "year": [
        "2019", "2020", "2021",
        "2022"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "zip",
    "variable": [
        "2m_temperature",
        "total_precipitation"
    ],
    "area": [40, -180, -40, 180]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
