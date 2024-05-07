import pandas as pd
import numpy as np
import random
import tensorflow as tf
from utils.utils import set_seeds, load_config
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import sys

class PortfolioBaseNN:
    def __init__(self):
        pass

    def build_model(self):
        pass


# custom loss function for gradient ascent Sharpe Ratio
def negative_sharpe_loss(wts, rets):
  mean_return = tf.reduce_mean(tf.reduce_sum(wts * rets, axis=1))
  std_return = tf.math.reduce_std(tf.reduce_sum(wts * rets, axis=1))

  return -mean_return / std_return


# Define MLP model
def create_MLP_model(custom_loss, asset_num, input_shape, params):

  model = tf.keras.models.Sequential()
  for i in range(params['num_hidden_layers']):
    if i==0:
      model.add(tf.keras.layers.Dense(units=params['hidden_layer_sizes'], activation=params['activation'], input_shape=(input_shape,)))
    else:
      model.add(tf.keras.layers.Dense(units=params['hidden_layer_sizes'], activation=params['activation']))

    if params['dropout_rate'] > 0.0:
      model.add(tf.keras.layers.Dropout(params['dropout_rate']))
  model.add(tf.keras.layers.Dense(units=asset_num, activation="softmax")) # <- softmax creates long-only portfolios

  optimizer_instance = tf.keras.optimizers.get(params['optimizer'])
  optimizer_instance.learning_rate = params['learning_rate']

  model.compile(optimizer=optimizer_instance, loss=custom_loss)

  return model


def walk_forward(n_train, n_val, df, epochs, batch, model, asset_num, best_params, best_score):
  n_splits = (df.shape[0] - n_train) // n_val + 1
  avg_score = 0.0

  preds = []

  for i in range(0, df.shape[0] - n_train, n_val):

    X_train, y_train = df.iloc[i : i + n_train, :-asset_num], df.iloc[i : i + n_train, -asset_num:]
    X_val, y_val = df.iloc[i + n_train : i + n_train + n_val, :-asset_num], df.iloc[i + n_train : i + n_train + n_val, -asset_num:]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch,
        validation_data = (X_val, y_val),
        verbose=1
    )

    y_pred = model.predict(X_val, verbose=0)
    preds.append(y_pred)

    score = model.evaluate(X_val, y_val, verbose=0)
    avg_score += score / n_splits

  return preds, avg_score



def main():
    # load config to get params
    config = load_config()
    trainval_split = float(config.get('NeuralNetParams', 'trainval_split'))
    asset_num = int(config.get('MetaData', 'ticker_num'))
    n_train = int(config.get('NeuralNetParams', 'n_train'))
    n_val = int(config.get('NeuralNetParams', 'n_val'))
    batch = int(config.get('NeuralNetParams', 'batch'))

    # load master dataset
    df_pivot = pd.read_csv('../data/final_dataset.csv', index_col='Date')
    df_pivot.index = pd.to_datetime(df_pivot.index)
    ticker_list = [i.split('ret_')[1] for i in df_pivot.columns[-asset_num:]]

    # let's establish train, validation, and test periods here:
    trainval_date_index = df_pivot.index[:int(trainval_split*df_pivot.shape[0])]
    test_date_index = df_pivot.index[int(trainval_split*df_pivot.shape[0]):]


    # need to scale our dataset
    df_pivot.reset_index(inplace=True, drop=True)

    # let's re-scale all of the data points
    scaler = MinMaxScaler()
    df_pivot_scaled = pd.DataFrame(scaler.fit_transform(df_pivot))

    # now set test dataset aside for post-training evaluation
    df_trainval = df_pivot_scaled.iloc[:int(trainval_split*df_pivot.shape[0])]
    df_test = df_pivot_scaled.iloc[int(trainval_split*df_pivot.shape[0]):]
    

    mlp_input_shape = df_trainval.iloc[:, :-asset_num].shape[1]

    param_grid = {
            'hidden_layer_sizes': [64, 128, 256],
            'num_hidden_layers': [1, 2],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'sgd'],
            'learning_rate': [0.001, 0.01, 0.1],
            'epochs': [50, 100],
            'dropout_rate': [0.0, 0.1, 0.2]
    }


    best_score = np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        print("Testing parameters: ", params)

        mlp = create_MLP_model(negative_sharpe_loss,
                                asset_num,
                                mlp_input_shape,
                                params)

        preds, avg_score = walk_forward(n_train=n_train,
                                            n_val=n_val,
                                            df=df_trainval,
                                            epochs=params['epochs'],
                                            batch=batch,
                                            model=mlp,
                                            asset_num=asset_num,
                                            best_params=best_params,
                                            best_score=best_score)

        if avg_score < best_score:
            best_score = avg_score
            best_params = params


    # train with best params
    mlp = create_MLP_model(negative_sharpe_loss,
                            asset_num,
                            mlp_input_shape,
                            best_params)

    mlp_history = mlp.fit(
                        df_trainval.iloc[:, :-asset_num],
                        df_trainval.iloc[:, -asset_num:],
                        epochs=best_params['epochs'],
                        batch_size=batch,
                        verbose=1
                )
    

    y_pred_test_mlp = mlp.predict(df_test.iloc[:, :-asset_num])

    mlp_wts = pd.DataFrame(y_pred_test_mlp, index=test_date_index, columns=ticker_list)

    mlp.save("../data/models/mlp")

    mlp_wts.to_csv("../data/models/mlp_wts.csv")



if __name__=="__main__":
   sys.exit(main())