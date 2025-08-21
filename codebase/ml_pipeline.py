from datetime import date

import numpy as np
import pandas as pd

# from tensorflow.keras.models import Model


def LSTM_preprocessing_nh(
    grdc_id: int,
    grdc_sub_ids: list | None,  ## MUST BE ORDERED DOWNSTREAM (first) TO UPSTREAM (last)
    dam_name: str,
    start_year: int,
    stop_year_ex: int,
    basin_str: str,
    grdc_dir: str = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/",
    met_dir: str = "/global/scratch/users/ann_scheliga/era5_data/",
    res_dir: str = "/global/scratch/users/ann_scheliga/CYGNSS_daily/",
    basin_data_dir: str = "/global/scratch/users/ann_scheliga/basin_forcing_processed/",
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Subset and consolidate meteorological, surface water, and streamflow daily
     time series for neuralhydrology LSTM input.

    Long Description
    ----------------

    Inputs
    ------
    grdc_id: int
        GRDC ID of the main (largest) basin
    grdc_sub_ids: list | None
        list must be in downstream (largest) to upstream (smallest)
        GRDC IDs of stream gauges further upstream
    dam_name: str
        name of dam as appears in GRanD database
    start_year: int
        time series starts on the first day of this year
    stop_year_ex: int
        time series ends on the first day of this year
    grdc_dir: str
        default = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/"
        where GRDC streamflow data stored
    met_dir: str
        default = "/global/scratch/users/ann_scheliga/era5_data/"
        where semi-global, annual ERA5 met data stored
    res_dir: str
        default = "/global/scratch/users/ann_scheliga/CYGNSS_daily/"
        where reservoir daily time series stored
    basin_data_dir: str
        default = "/global/scratch/users/ann_scheliga/basin_forcing_processed/"
        where output will be stored
    save_output: bool
        default = True
        whether to write the consolidated dataframe as a neuralhydrology-ready .pkl

    Outputs
    -------
    output_df : pd.DataFrame
    """
    import pickle

    import codebase

    if grdc_sub_ids is None:
        grdc_sub_ids = []

    ## Create output dataframe
    full_time = pd.date_range(
        start=date(start_year, 1, 1), end=date(stop_year_ex, 1, 1), freq="D"
    )
    output_df = pd.DataFrame(index=full_time)

    ## import sw_area
    sw_area = codebase.load_data.load_daily_reservoir_CYGNSS_area(
        dam_name, filepath=res_dir
    )
    output_df["SW_area"] = sw_area

    ## Calculate SW_flag
    output_df["SW_flag"] = (~output_df["SW_area"].isna()).astype(int)
    # where SW_area has a value, SW_flag is true

    ## Import basin watershed
    ## import GRDC
    watershed_gpd, grdc_Q = codebase.load_data.load_GRDC_station_data_by_ID(
        grdc_id,
        filepath=grdc_dir,
        timeseries_dict={"start_year": start_year, "stop_year": stop_year_ex},
        basin_str=basin_str,
    )
    grdc_Q.replace(-999, np.nan, inplace=True)
    output_df["Q"] = grdc_Q

    ## Get subbasin geometries
    subbasins_GRDC = [
        codebase.load_data.load_GRDC_station_data_by_ID(
            grdc_id,
            filepath=grdc_dir,
            timeseries_dict={"start_year": start_year, "stop_year": stop_year_ex},
            basin_str=basin_str,
        )
        for grdc_id in grdc_sub_ids
    ]
    # drop flow timeseries tuple from list, leave just the geoDataFrame(s)
    subbasin_shps = [output[0] for output in subbasins_GRDC]

    ## Create exclusive area geometries
    XOR_geoms = codebase.area_subsets.create_XOR_subasins(subbasin_shps, watershed_gpd)

    ## Consolidate overlapping area geometries
    subsets_gpd = pd.concat([watershed_gpd, *subbasin_shps])
    subsets_geoms = subsets_gpd["geometry"].reset_index(drop=True).add_prefix("_tot")

    ## Consolidate all (sub)basin geometries
    all_shps = pd.concat([subsets_geoms, XOR_geoms]).rename("geometry")

    ## Load met data of all geometries
    # Consider adding Pool.map() for parallelization
    met_list = [
        codebase.load_data.add_era5_met_data_by_shp(
            all_shps.loc[[idx]], filepath=met_dir, col_suffix=idx
        )
        for idx in all_shps.index
    ]
    met_df = pd.concat(met_list, axis=1)
    output_df = output_df.join(met_df, how="left")

    ## Final formatting
    output_df["SW_area"].fillna(output_df["SW_area"].mean(), inplace=True)
    output_df.interpolate(
        method="linear", axis=0, inplace=True, limit=7
    )  # interpolate missing interior values
    output_df.index.name = "date"

    if save_output:
        output_dict = {grdc_id: output_df}
        filename = (str(grdc_id) + "_" + dam_name.lower() + ".pkl").replace(" ", "_")
        pickle.dump(output_dict, open(basin_data_dir + filename, "wb"))
        print(".pkl output saved in", basin_data_dir, "as", filename)

    return output_df


def LSTM_preprocessing(
    res_name: str,
    dam_name: str,
    grdc_id: int,
    grdc_dir: str = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/",
    met_dir: str = "/global/scratch/users/ann_scheliga/era5_test_data/",
    res_dir: str = "/global/scratch/users/ann_scheliga/CYGNSS_daily/",
) -> pd.DataFrame:
    import pandas as pd

    import codebase

    ## import sw_area
    sw_area = codebase.load_data.load_daily_reservoir_CYGNSS_area(
        dam_name, filepath=res_dir
    )

    ## import GRDC
    watershed_gpd, grdc_Q = codebase.load_data.load_GRDC_station_data_by_ID(
        grdc_id,
        filepath=grdc_dir,
        timeseries_dict={"start_year": 2019, "stop_year": 2024},
    )

    ## import era5 met data
    tempK_files = codebase.utils.grab_matching_names_from_filepath(
        met_dir, r"daily_tempK"
    )
    precip_files = codebase.utils.grab_matching_names_from_filepath(
        met_dir, r"daily_tot_precip"
    )
    # type_precip_files = codebase.utils.grab_matching_names_from_filepath(
    #     met_dir,r'daily_precip_type'
    # )

    concat_dict = {"dim": "valid_time"}

    tempK_xr = codebase.area_subsets.era5_shape_subset_and_concat(
        ordered_filenames=tempK_files,
        filepath=met_dir,
        concat_dict=concat_dict,
        subset_gpd=watershed_gpd,
    )
    tempK_1dim = codebase.area_calcs.CYGNSS_001_areal_average(
        tempK_xr, x_dim="longitude", y_dim="latitude", with_index="valid_time"
    )
    tempK_1dim.rename("Temp K", inplace=True)
    precip_xr = codebase.area_subsets.era5_shape_subset_and_concat(
        ordered_filenames=precip_files,
        filepath=met_dir,
        concat_dict=concat_dict,
        subset_gpd=watershed_gpd,
    )
    precip_1dim = codebase.area_calcs.CYGNSS_001_areal_aggregation(
        np.nansum,
        precip_xr,
        x_dim="longitude",
        y_dim="latitude",
        with_index="valid_time",
    )
    precip_1dim.rename("Precip m", inplace=True)
    # type_precip_xr = codebase.area_subsets.era5_shape_subset_and_concat(
    #     ordered_filenames=type_precip_files,
    #     filepath=met_dir,
    #     concat_dict=concat_dict,
    #     subset_gpd=watershed_gpd,
    # )
    ## END impot era5 data

    ## aggregate data
    all_data = pd.concat([tempK_1dim, precip_1dim, sw_area, grdc_Q], axis=1)
    all_data.interpolate(
        method="linear", axis=0, inplace=True, limit=7
    )  # interpolate missing interior values
    all_data.bfill(inplace=True, limit=2)  # backfill missing first precip value

    return all_data


def split_data_and_reshape(all_data: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Split into X & y and train & test. Reshapes to 3D for input into LSTM."""
    from sklearn.model_selection import train_test_split

    X = all_data.drop(columns=["Q m3s"])[:-6].values
    y = all_data["Q m3s"][:-6].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=23, shuffle=False
    )
    X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    y_train = y_train.reshape(1, y_train.shape[0], 1)
    y_test = y_test.reshape(1, y_test.shape[0], 1)
    print(f"X train shape: {X_train.shape}; y train shape: {y_train.shape}")
    print(f"X test shape: {X_test.shape}; y test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def met_split(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, ...]:
    """Remove the last column of data, where SW area is stored."""
    X_met_train = X_train[:, :, :-1].copy()
    X_met_test = X_test[:, :, :-1].copy()
    return X_met_train, X_met_test


# def make_LSTM_1layer_model(n_timesteps_in: int, n_features: int) -> Model:
#     """
#     Create a keras model.
#     Stores the model as a function, so all experiments get the same model.
#     """
#     from tensorflow.keras import Input
#     from tensorflow.keras.layers import (
#         LSTM,
#         Dense,
#     )
#     from tensorflow.keras.models import Sequential

#     model = Sequential()
#     model.add(Input(shape=(n_timesteps_in, n_features)))
#     model.add(LSTM(150, dropout=0.2))
#     model.add(Dense(n_timesteps_in, activation="relu"))
#     model.compile(
#         loss="MeanSquaredError", optimizer="adam", metrics=["MeanAbsoluteError"]
#     )
#     print(model.summary())
#     return model


# def make_LSTM_2layer_model(n_timesteps_in: int, n_features: int) -> Model:
#     """
#     Create a keras model.
#     Stores the model as a function, so all experiments get the same model.
#     """
#     from tensorflow.keras import Input
#     from tensorflow.keras.layers import (
#         LSTM,
#         Dense,
#         RepeatVector,
#     )
#     from tensorflow.keras.models import Sequential

#     model = Sequential()
#     model.add(Input(shape=(n_timesteps_in, n_features)))
#     # model.add(Dense(128, activation='sigmoid'))
#     model.add(LSTM(150, dropout=0.2))
#     model.add(RepeatVector(n_timesteps_in))
#     model.add(LSTM(150, return_sequences=False))
#     # model.add(TimeDistributed(Dense(n_features, activation='softmax')))

#     model.add(Dense(n_timesteps_in, activation="relu"))
#     model.compile(
#         loss="MeanSquaredError", optimizer="adam", metrics=["MeanAbsoluteError"]
#     )
#     print(model.summary())
#     return model


def compare_epoch_error(
    history_nw: dict,
    history_sw: dict,
    error_metric: str = "MeanAbsoluteError",
    fig_name: str = "",
    legend: bool = True,
) -> None:
    """
    Plot training and val error for two models.

    Long Description
    ----------------
    Labels are set up for met-only vs met+SW area model comparison.
    Saves the plot if a fig_name is provided.

    Inputs
    ------
    history_nw, history_sw : dict
        dictionary from keras History.history
        History object is output from fitting a model.
        _nw for "no water"
        _sw for "surface water"
    error_metric : str
        key for history dictionaries
    fig_name : str
        default = ""
        if string is given, saves the plot in ../figures/
        as fig_name.png
    legend : bool
        default = True
        whether to include a legend

    Outputs
    -------
    None
    """
    import matplotlib.pyplot as plt

    error_nw = history_nw[error_metric]
    val_error_nw = history_nw["val_" + error_metric]
    error_sw = history_sw[error_metric]
    val_error_sw = history_sw["val_" + error_metric]
    epochs = range(1, len(error_nw) + 1)
    plt.plot(
        epochs,
        error_nw,
        color="navy",
        linestyle=":",
        alpha=0.8,
        label="Training error met only",
    )
    plt.plot(
        epochs,
        val_error_nw,
        color="green",
        linestyle=":",
        alpha=0.8,
        label="Validation error met only",
    )
    plt.plot(
        epochs,
        error_sw,
        color="navy",
        linestyle="-",
        alpha=0.8,
        label="Training error with SW",
    )
    plt.plot(
        epochs,
        val_error_sw,
        color="green",
        linestyle="-",
        alpha=0.8,
        label="Validation error with SW",
    )
    plt.title("Training and validation" + error_metric)
    plt.xlabel("Epochs")
    plt.ylabel(error_metric)
    if legend:
        plt.legend()
    if fig_name:
        plt.savefig("../figures/" + fig_name + ".png")
    return plt.show()
