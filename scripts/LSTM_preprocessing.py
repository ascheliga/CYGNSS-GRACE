from codebase import ml_pipeline

## Experimental set-up
grdc_id = 4152450
grdc_sub_ids = [
    4152600
]  # [4152450,4152600] ## MUST BE ORDERED DOWNSTREAM (first) TO UPSTREAM (last)
dam_name = "glen canyon"
start_year = 2018
stop_year_ex = 2024
basin_str = "colorado"

## Filepaths
grdc_dir = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/"
met_dir = "/global/scratch/users/ann_scheliga/era5_data/"
res_dir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/"
basin_data_dir = "/global/scratch/users/ann_scheliga/basin_forcing_processed/"

output_df = ml_pipeline.LSTM_preprocessing_nh(
    grdc_id,
    grdc_sub_ids,
    dam_name="glen canyon",
    start_year=2018,
    stop_year_ex=2024,
    basin_str=basin_str,
    save_output=True,
)

print("------- DESCRIPTION -------", flush=True)
print(output_df.describe(), flush=True)
print("------- SHAPE -------", flush=True)
print(output_df.shape, flush=True)
