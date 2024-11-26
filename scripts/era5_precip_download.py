import cdsapi

dataset = "insitu-gridded-observations-global-and-regional"
request = {
    "origin": "gpcc",
    "region": "global",
    "variable": ["precipitation"],
    "time_aggregation": "daily",
    "horizontal_aggregation": ["1_x_1"],
    "year": [
        "2018", "2019", "2020",
        "2021"
    ],
    "version": ["v2020_0_v6_0_fg"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()