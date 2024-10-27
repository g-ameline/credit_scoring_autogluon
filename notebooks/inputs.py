import numpy
import day
import price
import strat
import constant

nominal_row_length = 300 # price and date inputs

def price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(
    data_frame, 
    nominal_row_length=nominal_row_length,
):
    data_frame = data_frame.sort_values(by=constant.ColumnNames.ticker).reset_index(drop = True)
    # data_frame = strat.past_tickers_removed_data_frame_from_from_data_frame(data_frame)
    ticker_tensor  = numpy.sort(data_frame[constant.ColumnNames.ticker].unique())
    assert ticker_tensor[0] == data_frame[constant.ColumnNames.ticker].unique()[0],\
        f"{ticker_tensor[0]  = } {data_frame[constant.ColumnNames.ticker].unique()[0] = }"
    assert ticker_tensor[-1] == data_frame[constant.ColumnNames.ticker].unique()[-1],\
        f"{ticker_tensor[-1]  = } {data_frame[constant.ColumnNames.ticker].unique()[-1] = }"
    assert ticker_tensor[-1] == numpy.unique(data_frame[constant.ColumnNames.ticker].values)[-1],\
        f"{ticker_tensor[-1]  = } {numpy.unique(data_frame[constant.ColumnNames.ticker].values)[-1] = }"
    assert ticker_tensor[0] == numpy.unique(data_frame[constant.ColumnNames.ticker].values)[0],\
        f"{ticker_tensor[0]  = } {numpy.unique(data_frame[constant.ColumnNames.ticker].values)[0] = }"
    price_tensor = numpy.empty(
        shape= (len(ticker_tensor), nominal_row_length), 
        dtype=price.price_type,
    )
    latest_day_time_64 = numpy.sort(numpy.unique(data_frame[constant.ColumnNames.date].values))[-1]
    day_time_64_tensor = numpy.empty(
        shape=(len(ticker_tensor), nominal_row_length), 
        dtype='datetime64[D]',
    )
    for ticker_index, ticker in enumerate(ticker_tensor):
        ticker_data_frame = strat.ticker_data_from_data_frame_and_ticker_name(
            data_frame, ticker
        ).sort_values(by=constant.ColumnNames.date).reset_index(drop = True)
        ticker_day_time_64s = ticker_data_frame[constant.ColumnNames.date].values
        assert ticker_day_time_64s[-1] == latest_day_time_64, f"{ticker_day_time_64s[-1] = }{latest_day_time_64 = }"
        ticker_prices = ticker_data_frame[constant.ColumnNames.price].values
        assert len(ticker_day_time_64s) == len(ticker_prices), \
            f"{len(ticker_day_time_64s) = } {len(ticker_prices) = }"
        row_length = len(ticker_prices)
        if nominal_row_length > row_length:
            resized_ticker_prices = numpy.pad(
                array=ticker_prices,
                pad_width=(nominal_row_length - row_length,0),
                mode='constant',
                constant_values=numpy.nan,
                # constant_values=numpy.float32(0),
            )
            resized_ticker_day_time_64s = numpy.pad(
                array=ticker_day_time_64s,
                pad_width=(nominal_row_length - row_length, 0),
                mode='constant',
                # constant_values=constant.epoch_day_time_64,
                constant_values=numpy.datetime64("NaT"),
            )
        if nominal_row_length < row_length:
            resized_ticker_prices = ticker_prices[-nominal_row_length:]
            resized_ticker_day_time_64s = ticker_day_time_64s[-nominal_row_length:]
        if nominal_row_length == row_length:
            resized_ticker_prices = ticker_prices 
            resized_ticker_day_time_64s = ticker_day_time_64s
        price_tensor[ticker_index] = resized_ticker_prices
        day_time_64_tensor[ticker_index] = resized_ticker_day_time_64s
        assert len(resized_ticker_day_time_64s) == len(resized_ticker_prices) == nominal_row_length
    assert len(numpy.unique(day_time_64_tensor[:,-1])) == 1, f"{numpy.unique(day_time_64_tensor[:,-1]) = } {latest_day_time_64 = }"
    return price_tensor, day_time_64_tensor, ticker_tensor 

def resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
        subset_price_tensor,
        subset_day_time_64_tensor,
        subset_ticker_tensor, 
        superset_ticker_tensor, 
    ):
    for day_time_64s in subset_day_time_64_tensor:
        assert not numpy.isnat(day_time_64s[-1]), f"{day_time_64s = }"
    resized_subset_price_tensor = numpy.empty(
        shape=(len(superset_ticker_tensor), subset_price_tensor.shape[1]) ,
        # fill_value='',
        dtype=subset_price_tensor.dtype
    )
    min_day_time_64 = numpy.min(subset_day_time_64_tensor[~numpy.isnat(subset_day_time_64_tensor)])
    resized_subset_day_time_64_tensor = numpy.full(
        shape=( len(superset_ticker_tensor), subset_day_time_64_tensor.shape[1] ),
        fill_value=min_day_time_64,
        # fill_value=numpy.min(subset_day_time_64_tensor),
        # fill_value=constant.epoch_day_time_64,
        # fill_value=numpy.datetime64("NaT"),
        dtype='datetime64[D]',
        # dtype=subset_day_time_64_tensor.dtype,
    )
    resized_subset_ticker_tensor = numpy.empty_like(superset_ticker_tensor)
    subset_index = 0
    superset_index = 0
    while True:
        if subset_index == len(subset_ticker_tensor) and superset_index == len(superset_ticker_tensor):
            break
        if subset_index >= len(subset_ticker_tensor) or superset_index >= len(superset_ticker_tensor):
            raise Exception(f"{subset_index = } {superset_index = }")
        assert (
            not (subset_index == len(subset_ticker_tensor) or superset_index == len(superset_ticker_tensor))
            , f"{subset_index = } {superset_index = }"
        )            
        if subset_ticker_tensor[subset_index] != superset_ticker_tensor[superset_index]:
            superset_index+=1
            continue
        if subset_ticker_tensor[subset_index] == superset_ticker_tensor[superset_index]:
            resized_subset_price_tensor[superset_index] = subset_price_tensor[subset_index]
            resized_subset_day_time_64_tensor[superset_index] = subset_day_time_64_tensor[subset_index]
            resized_subset_ticker_tensor[superset_index] = subset_ticker_tensor[subset_index]
            subset_index+=1
            superset_index+=1
            continue
        raise Exception(f"{subset_index = } {superset_index = }")

    assert (
        numpy.isin(superset_ticker_tensor, subset_ticker_tensor).sum() ==
        numpy.isin(resized_subset_ticker_tensor, subset_ticker_tensor).sum() # ==
    ), f"{resized_subset_day_time_64_tensor = }"
    for day_time_64s in resized_subset_day_time_64_tensor:
        assert not numpy.isnat(day_time_64s[-1]), f"{day_time_64s = }"
    return (
        resized_subset_price_tensor, 
        resized_subset_day_time_64_tensor, 
        resized_subset_ticker_tensor,
    )

def next_and_next_next_delta_day_tensor_from_day_time_64_tensor(day_time_64_tensor):
    assert numpy.issubdtype(day_time_64_tensor.dtype, numpy.datetime64), f"{day_time_64_tensor.dtype = }"
    number_of_rows = len(day_time_64_tensor)
    last_day_time_64 = day_time_64_tensor[0][-1]
    next_and_next_day_delta_tensor = numpy.empty(
        (number_of_rows, 2),
        dtype=day.delta_day_type,
    )
    opened_dat_time_64s =  day.get_opened_day_time_64s()
    for day_time_row_index, day_time_row in enumerate(day_time_64_tensor):
        if numpy.isnat(day_time_row[-1]):
            next_and_next_day_delta_tensor[day_time_row_index] = numpy.array([0,0])            
            continue
        
        if day_time_row[-1] not in opened_dat_time_64s:
            raise Exception(f"{day_time_row = }")
        next_day_time_64, next_next_day_time_64 =  day.two_next_opened_trading_day_time_64(day_time_row[-1])
        as_datetime64Ds = numpy.array(
            [next_day_time_64, next_next_day_time_64],
            dtype='datetime64[D]',
        )
        as_deltatime64Ds = as_datetime64Ds - day_time_row[-1]
        next_and_next_day_delta_tensor[day_time_row_index] = (
            day.delta_day_type(as_deltatime64Ds / numpy.timedelta64(1, 'D'))
        )
    return next_and_next_day_delta_tensor/5

def aposteriori_delta_day_tensor_from_day_time_64_tensor(day_time_64_tensor):
    aposteriori_delta_day_tensor = numpy.empty(
        shape= day_time_64_tensor.shape,
        dtype=day.delta_day_type,
    )
    for row_index, day_time_64s in enumerate(day_time_64_tensor):
        aposteriori_delta_day_tensor[row_index] = day.aposteriori_delta_days_from_date_time_64s(day_time_64s)
        assert not any(numpy.isnan(aposteriori_delta_day_tensor[row_index])), f"{day_time_64s = } {aposteriori_delta_day_tensor[row_index]  = }"
    return aposteriori_delta_day_tensor

def aposteriori_relative_price_tensor_from_price_tensor(price_tensor):
    return numpy.stack(
        [
            price.aposteriori_relative_prices_from_prices(prices) if prices[-1] != 0 else prices
            for prices in price_tensor
        ], 
    )

def delta_of_relative_price_tensor_from_price_tensor(price_tensor):
    # for prices in price_tensor:
    #     assert not any([numpy.isnan(price) for price in prices ])
    # for prices in price_tensor:
    #     assert not any([price == 0 for price in prices ])
    delta_of_relative_price_tensor =  numpy.stack(
        [
            numpy.diff(prices)
            for prices in price_tensor
        ], 
    )/price_tensor[:,:-1]
    for i, d_rel_prices in enumerate(delta_of_relative_price_tensor):
        for j, d_rel_price in enumerate(d_rel_prices):
            if numpy.isnan(d_rel_price) :
                delta_of_relative_price_tensor[i][j] = 0 
    for prices in delta_of_relative_price_tensor:
        assert not any([numpy.isnan(price) for price in prices ]), f"{prices = }"
    for prices in delta_of_relative_price_tensor:
        assert not any([numpy.isnan(price) for price in prices ])
    return delta_of_relative_price_tensor

def scaled_aposteriori_relative_price_tensor_from_price_tensor(price_tensor):
    aposteriori_relative_price_tensor =  numpy.stack(
        [
            price.aposteriori_relative_prices_from_prices(prices) if prices[-1] != 0 else prices
            for prices in price_tensor
        ], 
    )
    average = numpy.average(aposteriori_relative_price_tensor) 
    deviation = numpy.std(aposteriori_relative_price_tensor) 
    scaled_aposteriori_relative_price_tensor = (aposteriori_relative_price_tensor - average)/deviation

def last_price_tensor_from_price_tensor(price_tensor):
    return numpy.array(
        [ [prices[-1]] for prices in price_tensor ]
    )

def ticker_one_hot_encoded_tensor_from_ticker_tensor_and_superset_ticker_tensor(
        ticker_tensor, superset_ticker_tensor
    ):
    assert len(set(superset_ticker_tensor)) == len(superset_ticker_tensor)
    assert numpy.all(numpy.sort(superset_ticker_tensor ) == superset_ticker_tensor)
    ticker_name_to_ticker_index = {
        ticker_name: index for index, ticker_name in enumerate(superset_ticker_tensor)
    }
    one_hot_encoded = numpy.zeros((len(ticker_tensor), len(superset_ticker_tensor )))
    for row, ticker_name in enumerate(ticker_tensor):
        if ticker_name is not None:
            one_hot_encoded[row, ticker_name_to_ticker_index[ticker_name]] = 1
            continue
        if ticker_name is None:
            continue
        1/0
    assert all(0<= ohe_row.sum() <=1 for ohe_row in  one_hot_encoded)
    assert one_hot_encoded.sum() == (ticker_tensor!=None).sum(), f"{one_hot_encoded.sum() = } {(ticker_tensor!=None).sum() = } {len(ticker_tensor) = } {len(superset_ticker_tensor) = }" 
    assert one_hot_encoded.sum() <= len(one_hot_encoded), f"{one_hot_encoded = } {ticker_tensor = }" 
    return one_hot_encoded

def different_inputs_from_data_frame_and_input_functions(
    data_frame, 
    superset_ticker_tensor,
    price_tensor_to_input_functions,
    day_time_64_tensor_to_input_functions,
    ticker_tensor_to_input_functions,
):
    price_tensor, day_time_64_tensor, ticker_tensor = (
        price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(data_frame)
    )
    price_tensor, day_time_64_tensor, ticker_tensor = (
        resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
            price_tensor, day_time_64_tensor, ticker_tensor,
            superset_ticker_tensor,
        )
    )
    for day_time_64s in day_time_64_tensor:
        assert not numpy.isnat(day_time_64s[-1]), f"{day_time_64s = }"
    inputs = []
    for input_from_price_tensor in price_tensor_to_input_functions:
        inputs.append(numpy.array([input_from_price_tensor(price_tensor)]))
        for row in inputs[-1]:
            assert not numpy.any(numpy.isnan(row)), f"{input_from_price_tensor = } {row = }"
    for input_from_day_time_64_tensor in day_time_64_tensor_to_input_functions:
        inputs.append(numpy.array([input_from_day_time_64_tensor(day_time_64_tensor)]))
        for row in inputs[-1]:
            assert not numpy.any(numpy.isnan(row)), f"{input_from_day_time_64_tensor = } {row = }"
    inputs.append(numpy.array([
        ticker_one_hot_encoded_tensor_from_ticker_tensor_and_superset_ticker_tensor(
            ticker_tensor, superset_ticker_tensor,
        )
    ]))
    for row in inputs[-1]:
        assert not numpy.any(numpy.isnan(row)), f"{ticker_one_hot_encoded_tensor_from_ticker_tensor_and_superset_ticker_tensor= } {row = }"
    for an_input in inputs:
        for row in an_input:
            assert not numpy.any(numpy.isnan(row)), f"{row= }"
    return inputs
        
def input_shapes_from_inputs(inputs):
    return [an_input.shape[1:] for an_input in inputs]

def return_output_target_from_input_data_frame_and_output_data_frame(
    input_data_frame, # up_to_last_day_price_tensor
    output_target_data_frame, # next_day_only_data_frame
    superset_ticker_tensor,
):
    last_day_tensor, last_day_time_64_tensor, last_day_ticker_tensor = (
        price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(
            data_frame=input_data_frame, 
            nominal_row_length=1,
        )
    )
    last_day_tensor, _, _ = (
        resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
            last_day_tensor, last_day_time_64_tensor, last_day_ticker_tensor,
            superset_ticker_tensor,
        )
    )
    next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor = (
        price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(
            data_frame=output_target_data_frame, 
            nominal_row_length=1,
        )
    )
    next_day_price_tensor, _, _ = (
        resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
            next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor,
            superset_ticker_tensor,
        )
    )
    assert len(next_day_price_tensor) == len(last_day_tensor)

    relative_next_day_price_tensor = numpy.zeros(
        shape=(len(next_day_price_tensor),1),
        dtype=price.price_type,
    )
    for index, (next_day_price, last_day_price) in enumerate(
        zip(last_day_tensor, next_day_price_tensor)
    ):
        if last_day_price != 0:
            relative_next_day_price_tensor[index][0] = last_day_price/next_day_price-1
    
    return numpy.array([relative_next_day_price_tensor]) 


def price_output_target_from_input_data_frame_and_output_data_frame(
    input_data_frame, # up_to_last_day_price_tensor
    output_target_data_frame, # next_day_only_data_frame
    superset_ticker_tensor,
):
    next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor = (
        price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(
            data_frame=output_target_data_frame, 
            nominal_row_length=1,
        )
    )
    next_day_price_tensor, _, _ = (
        resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
            next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor,
            superset_ticker_tensor,
        )
    )
    return numpy.array([next_day_price_tensor]) 

# def return_output_from_input_data_frame_and_output_data_frame(
#     input_data_frame, # last_day_price_tensor
#     output_data_frame, # next_day_price_tensor
#     superset_ticker_tensor,
# ):
#     up_to_last_day_price_tensor, input_day_time_64_tensor, input_ticker_tensor = (
#         price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(input_data_frame,1)
#     )
#     up_to_last_day_price_tensor, _, _ = (
#         resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
#             up_to_last_day_price_tensor, input_day_time_64_tensor, input_ticker_tensor,
#             superset_ticker_tensor,
#         )
#     )
#     next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor = (
#         price_tensor_and_day_time_64_tensor_and_tickers_tensor_from_data_frame(
#             data_frame=output_data_frame, 
#             nominal_row_length=1,
#         )
#     )
#     next_day_price_tensor, _, _ = (
#         resized_price_tensor_and_day_time_64_tensor_according_to_tickers_tensor(
#             next_day_price_tensor, output_day_time_64_tensor, output_ticker_tensor,
#             superset_ticker_tensor,
#         )
#     )
#     assert len(next_day_price_tensor) == len(up_to_last_day_price_tensor)
#     future_return_tensor = numpy.zeros(
#         shape=(len(next_day_price_tensor),1),
#         dtype=price.price_type,
#     )

#     for index, (next_day_price_tensor, next_day_price_tensor) in enumerate(
#         zip(next_day_price_tensor, next_day_price_tensor_tensor)
#     ):
#         if next_day_price_tensor != 0:
#             relative_next_day_price_tensor[index][0] = next_day_price_tensor[0]/next_day_price_tensor - 1
#             relative_next_day_price_tensor[index][1] = next_day_price_tensor[1]/next_day_price_tensor - 1
                
#     assert len(next_day_price_tensor) == len(up_to_last_day_tensor)
#     relative_next_day_price_tensor= numpy.zeros(
#         shape=(len(next_day_price_tensor),2),
#         dtype=price.price_type,
#     )
#     # for index, (next_day_price, last_day_price) in enumerate(zip(next_day_price_tensor, up_to_last_day_tensor)):
#     #         if last_day_price != 0:
#     #             relative_next_day_price_tensor[index][0] = last_day_price/next_day_price[0]-1
#     #             relative_next_day_price_tensor[index][1] = D1_and_D2_prices[0]/D1_and_D2_prices[1]-1
    
#     return numpy.array([relative_next_day_price_tensor]) 
        
