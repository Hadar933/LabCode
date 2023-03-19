from typing import List, Optional
import numpy as np
import utils as utils
import pandas as pd


class Encoder:
    def __init__(self):
        pass

    @staticmethod
    def encode_derivatives(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        if not cols:
            cols = df.columns
        time = df.index.to_series().diff().dt.total_seconds().values[:, None]

        col_dot_names = [f"{col}_dot" for col in cols]
        df[col_dot_names] = df[cols].diff() / time

        col_ddot_names = [f"{col}_ddot" for col in cols]
        df[col_ddot_names] = df[col_dot_names].diff() / time
        return df


if __name__ == '__main__':
    start_time = '2023-03-19 12:00:00'
    experiment_time = 1  # second
    end_time = pd.to_datetime(start_time) + pd.Timedelta(seconds=experiment_time)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1ms')
    time_values = (time_index - time_index[0]).astype('timedelta64[ms]').astype(float) / 1000
    A = 1.0
    f = 10.0
    phi = 0.0

    # Generate the sine wave
    df = pd.DataFrame(data={'sin': A * np.sin(2 * np.pi * f * time_values + phi)},
                      index=time_index)

    new_df = encode_derivatives(df)
