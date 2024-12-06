from codebase import ml_pipeline

## DEFINE INPUTS
res_name = "Mead"
dam_name = "hoover"
grdc_id = 4152103

n_epochs = 20
## END INPUTS

## LOAD DATA
all_data = ml_pipeline.LSTM_preprocessing(
    res_name=res_name, dam_name=dam_name, grdc_id=grdc_id
)

## SPLIT DATA
X_train, X_test, y_train, y_test = ml_pipeline.split_data_and_reshape(all_data)
X_met_train, X_met_test = ml_pipeline.met_split(X_train, X_test)

n_timesteps_in = X_train.shape[-2]
n_features = X_train.shape[-1]
n_timesteps_out = 1
## END SPLIT DATA

##
model_nw = ml_pipeline.make_LSTM_model(n_timesteps_in, 2)
model_nw_history = model_nw.fit(
    X_met_train,
    y_train,
    epochs=n_epochs,
    batch_size=38,
    validation_data=(X_met_test, y_test),
)

##
model_sw = ml_pipeline.make_LSTM_model(n_timesteps_in, 3)
model_sw_history = model_sw.fit(
    X_train, y_train, epochs=n_epochs, batch_size=38, validation_data=(X_test, y_test)
)

## PLOT ERROR
fig_name = res_name + "_MAE"
ml_pipeline.compare_epoch_error(
    model_nw_history.history, model_sw_history.history, fig_name=fig_name
)
