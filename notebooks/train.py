import strat
import constant
import day
import numpy

def nontest_and_test_data_frames_from_data_frame(data_frame):
    print(f"first test date {constant.test_start_day_time_64= }")
    return strat.splitees_data_frames_from_date(data_frame, constant.test_start_day_time_64)

def folds_data_frame_from_nontest_data_frame(nontest_data_frame, number_of_folds):
    date_set_sequence = day.day_time_64_unique_sequence_from_data_frame(nontest_data_frame)
    slicing_date_couples = list(
        (day.iso_day_from_day_time_64(period[0]), day.iso_day_from_day_time_64(period[-1]))
        for period 
        in numpy.array_split(date_set_sequence, number_of_folds)
    )
    assert len(slicing_date_couples) == number_of_folds, f"{len(slicing_date_couples) = } != {number_of_folds = }"
    for first_and_last_couple in slicing_date_couples:
         # strat.slice_data_frame_from_first_and_last_dates(nontest_data_frame, first_and_last_couple)
        train_data_frame, valdiate_data_frame = strat.slice_data_frame_from_first_and_last_dates(nontest_data_frame, first_and_last_couple)
        yield train_data_frame, valdiate_data_frame 
    return 

def train_and_validate_data_frames_from_nontest_data_frame(nontest_data_frame):
    date_set_sequence = day.day_time_64_unique_sequence_from_data_frame(nontest_data_frame)
    middle_day_time_64= date_set_sequence[len(date_set_sequence)//2]
    print(f"first valdiate date {middle_day_time_64 = }")
    assert isinstance(middle_day_time_64, numpy.datetime64), f"{type(splitting_day_time_64) = }"
    train_data_frame, validate_data_frame = strat.splitees_data_frames_from_date(nontest_data_frame, middle_day_time_64)
    return [
        strat.past_tickers_removed_data_frame_from_from_data_frame(train_data_frame),
        strat.past_tickers_removed_data_frame_from_from_data_frame(validate_data_frame),
    ]
    
def up_to_before_last_day_data_frame_and_last_day_data_frame_from_data_frame(data_frame):
    assert len(data_frame[constant.ColumnNames.ticker].unique()) > 1, f"{data_frame[constant.ColumnNames.ticker] = }"
    day_date_64_sequence = numpy.sort(data_frame[constant.ColumnNames.date].unique())
    print(f"up to before last day date: {day_date_64_sequence[0]} -> {day_date_64_sequence[-2]}")
    print(f"last day date: {day_date_64_sequence[-1]}")
    input_data_frame, output_data_frame =  strat.splitees_data_frames_from_date(data_frame, day_date_64_sequence[-1])
    return [
        strat.past_tickers_removed_data_frame_from_from_data_frame(input_data_frame),
        strat.past_tickers_removed_data_frame_from_from_data_frame(output_data_frame),
    ]
    

