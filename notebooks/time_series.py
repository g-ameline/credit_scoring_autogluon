import numpy 
import pandas

def timestamps(time_series_data_frame):
    return time_series_data_frame.index.get_level_values('timestamp')

def from_first_up_to_before_last_date_series(time_series_data_frame, last_date):
    date_index = timestamps(time_series_data_frame)
    last_date_mask = date_index <= last_date
    return time_series_data_frame[last_date_mask]

def replace_last_row_with_prediction(
    time_series_data_frame,
    last_date_timestamp,
    predictor,
):
    from_first_up_to_before_last_date = from_first_up_to_before_last_date_series(time_series_data_frame, last_date_timestamp)
    last_date_as_prediction = predictor.predict(time_series_data_frame)
    last_date_as_prediction.rename(
        columns={'mean': 'target'}
    )
    return pandas.concat( [
        from_first_up_to_before_last_date,
        last_date_as_prediction,
    ] )

def append_predicted_next_row(
    time_series_data_frame,
    predictor,
    predicted_column_name='mean',
    to_be_predicted_column_name='target',
):
    next_rows_prediction = predictor.predict(time_series_data_frame)
    next_rows_prediction[to_be_predicted_column_name] = next_rows_prediction[predicted_column_name]
    return pandas.concat( [
        time_series_data_frame,
        next_rows_prediction,
    ] )

def keep_only_two_latest_rows(time_series_data_frame):
    date_index = timestamps(time_series_data_frame)
    two_last_dates = numpy.sort(
        numpy.unique(date_index)
    )[-2:]
    two_last_dates_mask = (date_index == two_last_dates[0]) | (date_index == two_last_dates[1])
    return time_series_data_frame[two_last_dates_mask]

def return_series(price_time_series):
    return price_time_series.pct_change()

def one_day_before_shifted_series(price_time_series):
    index_one_day_bedore = price_time_series.groupby('item_id').shift(-1)
    return index_one_day_bedore
    
def calculate_future_return_series(price_time_series):
    return price_time_series.groupby('item_id').shift(-1) / price_time_series - 1

def get_best_return_(return_time_series):
        return return_time_series.max()

def pick_n_random_dates_from_time_series_data_frame(time_series_data_frame, n=30):
    timestamps(time_series_data_frame)
    return numpy.random.choice(timestamps(time_series_data_frame), size=n, replace=False, p=None)
