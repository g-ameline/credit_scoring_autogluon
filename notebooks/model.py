import keras
import numpy

batch_size = 32
epochs = 20

def inputs_and_concatenate_and_flatten_layers_from_input_shapes(input_shapes,input_names=None):
    input_layers = []
    for index, input_shape in enumerate(input_shapes):
        input_layers.append(keras.Input(
            shape=input_shape,
            # batch_size=batch_size,
            dtype=numpy.float32,
            # sparse=None,
            # batch_shape=None,
            name=input_names[index] if input_names else f"input_{index}",
            # tensor=None,
            # optional=False,
    ))
    concatenate_layer = keras.layers.Concatenate()(input_layers)
    flatten_layer = keras.layers.Flatten()(concatenate_layer)
    return input_layers, concatenate_layer, flatten_layer

def hidden_layers_from_inputer_layer(inputer_layer, unitses):
    for index,units in enumerate(unitses):
        inputer_layer=keras.layers.Dense(
            units, 
            activation='elu',
            # activation='relu',
            name=f"hidden_layer_{index}"
        )(inputer_layer)
    return inputer_layer

def reshaped_output_layer_from_inputer_layer(inputer_layer, output_shape):
    # output shape might something like (None|training_batch_size, number_of_tickers, len([D1,D2]) )
    # first output shape value is the number of inputs (batch size for example )
    assert output_shape[0] > 100, f"{output_shape = } 's first length should be represent the nubmer of ticker dimension"
    assert output_shape[1] == 1, f"{output_shape = } 's second length should represent the next day price dimension"
    flat_output_layer = keras.layers.Dense(
        units=output_shape[0]*output_shape[1], 
        # activation='elu', 
        # activation='relu', 
        # activation='sigmoid', 
        # activation='softsign', 
        # activation='leaky_relu', 
        activation='linear', 
        name='output_D1_and_D2_prices_from_who_knows_what_hidden_layers_figured_out'
    )(inputer_layer)
    reshaped_output_layer = keras.layers.Reshape(output_shape)(flat_output_layer )
    return reshaped_output_layer 

def output_layer_from_inputer_layer(inputer_layer, output_shape):
    # output shape might something like (None|training_batch_size, number_of_tickers, len([D1,D2]) )
    # first output shape value is the number of inputs (batch size for example )
    assert output_shape[1] > 100, f"{output_shape = } 's first length should be represent the nubmer of ticker dmension"
    assert output_shape[2] == 2, f"{output_shape = } 's second length should represent the D1 and D2 prices dmension"
    output_layer = keras.layers.Dense(
        units=output_shape[0]*output_shape[1], 
        activation='relu', 
        # activation='linear', 
        name='output_D1_and_D2_prices_from_who_knows_what_hidden_layers_figred_out'
    )(inputer_layer)
    return output_layer

def model_from_input_and_output_layers(input_layer, output_layer):
    model = keras.Model(input_layer, output_layer)
    # opt = keras.optimizers.Adam(0.0001, clipnorm=1.)
    model.compile(
        # optimizer=opt, 
        optimizer='adam', 
        # loss='mse', 
        # loss='mae', 
        # loss='mean_squared_logarithmic_error', 
        loss='huber', 
        # loss='mean_squared_logarithmic_error', 
        # metrics=['mape'],
        metrics=['mae'],
    )

    model.summary()
    return model

# def compiled_model_from_input_tensors_and_output_shape(
#     input_tensors,
#     hidden_layer_sized,
#     output_shape,
# ):

def fitted_model_from_model_and_train_IO_and_validate_IO(
    model, 
    fit_inputs,
    fit_target_outputs,
    validate_inputs,
    validate_target_outputs,
    epochs=epochs,
    verbose=1,
    batch_size=batch_size,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='mae', min_delta=0.005, verbose=1),
        keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_mae', min_delta=0.005, verbose=1),
        # keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='mape', verbose=1),
        # keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor='val_mape', verbose=1),
    ],
):
    history = model.fit(
        x=fit_inputs,
        y=fit_target_outputs,
        batch_size=batch_size,
        epochs=epochs,
        verbose="auto",
        callbacks=callbacks,
        validation_split=0.0,
        validation_data=(validate_inputs,validate_target_outputs), 
        shuffle=False,
        )
    score_by_metric = { metric: values[-1] for metric, values in history.history.items() }
    print(f"{score_by_metric = }")
    return model 
