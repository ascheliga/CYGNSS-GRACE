import pandas as pd


def usbr_dataprocessing(df: pd.DataFrame) -> pd.DataFrame:
    from codebase.utils import convert_to_num

    df = df.drop(columns="Location")
    df.dropna(axis=0, how="any", inplace=True)
    df = df[df["Timestep"] != "Timestep"]  # remove repeat header rows
    df["Variable"] = df["Parameter"] + " [" + df["Units"] + "]"
    df["Result"] = df["Result"].apply(convert_to_num)
    df_pivot = df.pivot(columns="Variable", index="Datetime (UTC)", values="Result")
    df_pivot.index = pd.to_datetime(df_pivot.index)
    return df_pivot
