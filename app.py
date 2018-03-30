from flask import Flask, request
from flask import jsonify, render_template
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
import keras


def plot_data_between_dates(start_date, end_date):
    data = get_data_between_dates(start_date, end_date)
    data.drop(['date'], axis=1, inplace=True)
    return data


def remove_comma(df: pandas.DataFrame):
    for column in df.columns:
        df[column].replace(regex=True, inplace=True, to_replace=r',', value='')


def preprocess_data(df_ld, df_td):
    df_ld.drop(['zone_id', 'year', 'month', 'day'], axis=1, inplace=True)
    df_td.drop(['station_id', 'year', 'month', 'day'], axis=1, inplace=True)
    df_ld.dropna(axis=0, how='any', inplace=True)
    df_td.dropna(axis=0, how='any', inplace=True)
    remove_comma(df_ld)
    df_ld = df_ld.apply(pandas.to_numeric)
    return df_ld, df_td


def scale_data(data):
    temp_data = data.copy()
    scaler = MinMaxScaler()
    temp_data[temp_data.columns] = scaler.fit_transform(temp_data[temp_data.columns])
    return temp_data


def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for window in range(len(series) - window_size):
        X.append(series[window:window + window_size])
        y.append(series.iloc[[window + window_size]])
    true_data = []
    for elem in X:
        temp_list = []
        for column in elem.columns:
            temp_list.append(elem[column])
        true_data.append(temp_list)
    X = true_data
    X = numpy.asarray(X)
    true_data = []
    for elem in y:
        true_data.append(numpy.asarray(elem))
    y = true_data
    y = numpy.asarray(y)
    y.shape = (len(y), 24)
    return X, y


def predict_future(model, input_data, num_days):
    output_list = []
    input_list = input_data
    predicted_data = input_data.copy()
    for _ in range(num_days):
        predicted_data = model.predict(input_list)
        input_list = numpy.delete(input_list[0], obj=0, axis=0)
        input_list = numpy.append(input_list, predicted_data, axis=0)
        input_list = numpy.asarray(numpy.reshape(input_list, (1, window_size, 24)))
        output_list.append(predicted_data)
    return output_list


def plot_future_prediction(outputs):
    i = 0
    plt.figure(figsize=(24, 24))
    for output in outputs:
        i += 1
        plt.subplot(8, 4, i)
        plt.plot(range(0, 24), output[0])
        plt.title("Day " + str(i))
        plt.tight_layout()
    plt.show()


def get_data_between_dates(start_date=(2004, 1, 2), end_date=(2004, 1, 6)):
    a = load_data_with_date[
        load_data_with_date['date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d")) < datetime(end_date[0],
                                                                                                       end_date[1],
                                                                                                       end_date[2])]
    b = a[a['date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d")) > datetime(start_date[0], start_date[1],
                                                                                       start_date[2])]
    return b


def get_per_hour_data(start_date=(2008, 6, 30), end_date=(2008, 7, 20)):
    if datetime(start_date[0], start_date[1], start_date[2]) > datetime.strptime(
            load_data_with_date.loc[load_data_with_date.index[-1]]['date'], "%Y-%m-%d"):
        input_data = X_test[-1]
        num_days = 15
    else:
        data = plot_data_between_dates(start_date, end_date)
        input_data = scaled_load_data.loc[data.index[0] - window_size + 1: data.index[0]]
        num_days = len(data.index) if len(data.index) < 15 else 15
    input_data = numpy.asarray(numpy.reshape(numpy.asarray(input_data), (1, window_size, 24)))
    outputs = predict_future(model, input_data, num_days)
    return outputs


def get_per_day_prediction_data(start_date=(2008, 6, 30), end_date=(2008, 7, 20)):
    outputs = get_per_hour_data(start_date, end_date)
    per_day = [numpy.sum(day_data) for day_data in outputs]
    st = datetime(start_date[0], start_date[1], start_date[2])
    end = datetime(end_date[0], end_date[1], end_date[2])
    i_date = st
    dates = []
    while i_date < end:
        dates.append((i_date).strftime('%Y-%m-%d'))
        i_date = i_date + timedelta(hours=24)
    dates.append(end.strftime('%Y-%m-%d'))
    return per_day, dates


def get_one_day_load_prediction(date=(2008, 6, 30)):
    load, date =get_per_day_prediction_data(start_date=date,end_date=date)
    return load[0]


zone_id = 2
load_data = pandas.read_csv('static/data/Load_history.csv')
load_data = load_data[load_data.zone_id == zone_id]
temperature_data = pandas.read_csv('static/data/temperature_history.csv')
load_dat = load_data[['year', 'month', 'day']]
load_data, temperature_data = preprocess_data(load_data, temperature_data)
load_data_with_date = load_data.copy()
date_str = load_dat['year'].map(str) + '-' + load_dat['month'].map(str) + '-' + load_dat['day'].map(str)
load_data_with_date['date'] = date_str
scaled_load_data = scale_data(load_data)
scaled_temparature_data = scale_data(temperature_data)
window_size = 50
X, y = window_transform_series(series=scaled_load_data, window_size=window_size)
train_test_split = int(numpy.ceil(4 * len(y) / float(5)))  # set the split point
# partition the training set
X_train = X[:train_test_split, :]
y_train = y[:train_test_split]
# keep the last chunk for testing
X_test = X[train_test_split:, :]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize]
X_train = numpy.asarray(numpy.reshape(X_train, (X_train.shape[0], window_size, 24)))
X_test = numpy.asarray(numpy.reshape(X_test, (X_test.shape[0], window_size, 24)))
y_train = numpy.asarray(y_train)
y_test = numpy.asarray(y_test)
numpy.random.seed(0)

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, dropout=0.15,
               recurrent_dropout=0.1))
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, dropout=0.15,
               recurrent_dropout=0.1))
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(24))
# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.load_weights('static/model_weights/best_RNN_weights.hdf5')

# ------------------------------------------
app = Flask(__name__)


@app.route('/forcasts/', methods=['GET'])
def forcasts():
    # start_date = (2006, 1, 2)
    # end_date = (2006, 1, 31)
    data, dates = get_per_day_prediction_data()
    next_day_load = get_one_day_load_prediction()
    return render_template('forcasts.html', load_data=zip(data, dates), next_day_load=next_day_load,next_day='2008-6-30')


@app.route('/forcast_range/', methods=['POST'])
def forcast_range():
    from_date = [int(i) for i in request.form['from_date'].split('-')]
    to_date = [int(i) for i in request.form['to_date'].split('-')]
    data, dates = get_per_day_prediction_data(from_date,to_date)
    next_day_load = get_one_day_load_prediction()
    return render_template('forcasts.html', load_data=zip(data, dates), next_day_load=next_day_load,next_day='2008-6-30')


@app.route('/home/', methods=['GET'])
def home():
    print("home")
    return render_template('home.html')

if __name__ == '__main__':
    app.run()
