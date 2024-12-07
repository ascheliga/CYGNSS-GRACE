import numpy as np
import pandas as pd
from tensorflow.keras.models import Model


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

    codebase.load_data.load_GRanD()

    ## import sw_area
    sw_area = pd.read_csv(res_dir + dam_name + "_area.csv", index_col=0)
    sw_area.index = pd.to_datetime(sw_area.index)
    sw_area = sw_area.loc[sw_area.index < pd.to_datetime("2024-01-01"), "Area m2"]
    sw_area = (sw_area / 10**6).rename("Area km2", inplace=True)

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
        method="linear", axis=0, inplace=True, limit = 7
    )  # interpolate missing interiror values
    all_data.bfill(inplace=True,limit=2)  # backfill missing first precip value

    return all_data


def split_data_and_reshape(all_data: pd.DataFrame) -> tuple[np.ndarray, ...]:
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
    X_met_train = X_train[:, :, :-1].copy()
    X_met_test = X_test[:, :, :-1].copy()
    return X_met_train, X_met_test


def make_LSTM_model(n_timesteps_in: int, n_features: int) -> Model:
    from tensorflow.keras import Input
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        RepeatVector,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Input(shape=(n_timesteps_in, n_features)))
    # model.add(Dense(128, activation='sigmoid'))
    model.add(LSTM(150, dropout=0.2))
    model.add(RepeatVector(n_timesteps_in))
    model.add(LSTM(150, return_sequences=False))
    # model.add(TimeDistributed(Dense(n_features, activation='softmax')))

    model.add(Dense(n_timesteps_in, activation="relu"))
    model.compile(
        loss="MeanSquaredError", optimizer="adam", metrics=["MeanAbsoluteError"]
    )
    print(model.summary())
    return model


def compare_epoch_error(
    history_nw: dict,
    history_sw: dict,
    error_metric: str = "MeanAbsoluteError",
    fig_name: str = "",
) -> None:
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
        label="Training MAE met only",
    )
    plt.plot(
        epochs,
        val_error_nw,
        color="green",
        linestyle=":",
        alpha=0.8,
        label="Validation MAE met only",
    )
    plt.plot(
        epochs,
        error_sw,
        color="navy",
        linestyle="-",
        alpha=0.8,
        label="Training MAE with SW",
    )
    plt.plot(
        epochs,
        val_error_sw,
        color="green",
        linestyle="-",
        alpha=0.8,
        label="Validation MAE with SW",
    )
    plt.title("Training and validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    if fig_name:
        plt.savefig("../figures/" + fig_name + ".png")
    return plt.show()
