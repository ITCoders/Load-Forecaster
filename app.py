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

load_data_with_date = pandas.DataFrame

def remove_comma(df: pandas.DataFrame):
    for column in df.columns:
        df[column].replace(regex=True, inplace=True, to_replace=r',', value='')


def preprocess_data(df_ld):
    df_ld.drop(['zone_id', 'year', 'month', 'day'], axis=1, inplace=True)
    df_ld.dropna(axis=0, how='any', inplace=True)
    df_ld.reset_index(drop=True, inplace=True)
    remove_comma(df_ld)
    df_ld = df_ld.apply(pandas.to_numeric)
    return df_ld


def scale_data(data):
    temp_data = data.copy()
    scaler = MinMaxScaler()
    temp_data[temp_data.columns] = scaler.fit_transform(temp_data[temp_data.columns])
    return temp_data



def get_raw_load_data():
    load_data = pandas.read_csv('static/data/Load_history.csv')
    return load_data


def get_preprocessed_load_data(zone_id):
    raw_load_data = get_raw_load_data()
    zone_load_data = raw_load_data[raw_load_data.zone_id==zone_id]
    preprocessed_load_data = preprocess_data(zone_load_data)
    preprocessed_load_data = scale_data(preprocessed_load_data)
    return preprocessed_load_data


def get_preprocessed_load_data_with_date(zone_id):
    raw_data = get_raw_load_data()
    load_data_with_date = raw_data.copy()
    load_data_with_date = load_data_with_date[load_data_with_date.zone_id==zone_id]
    load_data_with_date.dropna(axis=0, how='any', inplace=True)
    load_data_with_date.reset_index(drop=True, inplace=True)
    date_str = load_data_with_date['year'].map(str) + '-' + load_data_with_date['month'].map(str) + '-' + load_data_with_date['day'].map(str)
    load_data_with_date.drop(['zone_id', 'year', 'month', 'day'], axis=1, inplace=True)
    remove_comma(load_data_with_date)
    load_data_with_date = load_data_with_date.apply(pandas.to_numeric)
    scaled_preprocessed_load_data_with_date = scale_data(load_data_with_date)
    scaled_preprocessed_load_data_with_date['date'] = date_str
    #print(scaled_preprocessed_load_data_with_date)
    return scaled_preprocessed_load_data_with_date


def plot_data_between_dates(start_date, end_date):
    data = get_data_between_dates(start_date, end_date)
    data.drop(['date'], axis=1, inplace=True)
    return data


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


def load_previous_weights(model, path='static/model_weights/best_RNN_weights.hdf5'):
    model.load_weights(path)
    return model


def predict_performance(model, X):
    return model.predict(X)


def get_error(model, X, y):
    return model.evaluate(X, y)


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


def get_data_between_dates(start_date, end_date,zone_id=3):
    scaled_preprocessed_load_data_with_date = get_preprocessed_load_data_with_date(zone_id=zone_id)
    a = scaled_preprocessed_load_data_with_date[scaled_preprocessed_load_data_with_date['date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d")) < datetime(end_date[0], end_date[1], end_date[2])]
    b = a[a['date'].apply(lambda date: datetime.strptime(date, "%Y-%m-%d")) > datetime(start_date[0], start_date[1], start_date[2])]
    return b


def get_per_hour_data(start_date = (2006, 1, 2), end_date =(2006, 3, 20),zone_id=3):
    scaled_preprocessed_load_data_with_date = get_preprocessed_load_data_with_date(zone_id=zone_id)
    #print(scaled_preprocessed_load_data_with_date)
    if datetime(start_date[0], start_date[1], start_date[2]) > datetime.strptime(scaled_preprocessed_load_data_with_date.loc[scaled_preprocessed_load_data_with_date.index[-1]]['date'], "%Y-%m-%d"):
        input_data = X_test[-1]
        num_days = 15
    else:
        print('It entered here')
        data = plot_data_between_dates(start_date, end_date)
        scaled_preprocessed_load_data_with_date.drop(['date'], inplace=True, axis=1)
        input_data = scaled_preprocessed_load_data_with_date.loc[data.index[0] - window_size + 1: data.index[0]]
        num_days = len(data.index)
    input_data = numpy.asarray(numpy.reshape(numpy.asarray(input_data), (1, window_size, 24)))
    outputs = predict_future(model, input_data, num_days)
    return outputs

def get_per_day_prediction_data(start_date=(2008, 6, 30), end_date=(2008, 7, 20),zone_id=3):
    outputs = get_per_hour_data(start_date, end_date,zone_id=zone_id)
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


def get_one_day_load_prediction(date=(2008, 6, 30),zone_id=3):
    load, date =get_per_day_prediction_data(start_date=date,end_date=date,zone_id=zone_id)
    return load[0]


# ------------------------------------------
app = Flask(__name__)


@app.route('/forcasts/', methods=['GET'])
def forcasts():
    start_date = (2006, 1, 2)
    end_date = (2006, 1, 17)
    data, dates = get_per_day_prediction_data(start_date,end_date)
    next_day_load = get_one_day_load_prediction()
    return render_template('forcasts.html', load_data=zip(data, dates), next_day_load=next_day_load,next_day='2008-6-30')


@app.route('/forcast_range/', methods=['POST'])
def forcast_range():
    from_date = [int(i) for i in request.form['from_date'].split('-')]
    to_date = [int(i) for i in request.form['to_date'].split('-')]
    data, dates = get_per_day_prediction_data(from_date,to_date)
    next_day_load = get_one_day_load_prediction()
    return render_template('forcasts.html', load_data=zip(data, dates), next_day_load=next_day_load,next_day='2008-6-30')


@app.route('/forcast_one_day/', methods=['POST'])
def forcast_one_day():
    print(request)
    print(request.form)
    print(request.json)
    day = [int(i) for i in request.json['day'].split('-')]
    print(day)
    load = get_one_day_load_prediction(date=day)
    print(load)
    return jsonify(load=str(load))

@app.route('/home/', methods=['GET'])
def home():
    print("home")
    return render_template('home.html')

if __name__ == '__main__':
    window_size = 10
    X, y = window_transform_series(get_preprocessed_load_data(zone_id=3), window_size)
    train_test_split = int(numpy.ceil(4*len(y)/float(5)))   # set the split point
    X_train = X[:train_test_split,:]
    y_train = y[:train_test_split]
    X_test = X[train_test_split:,:]
    y_test = y[train_test_split:]
    X_train = numpy.asarray(numpy.reshape(X_train, (X_train.shape[0], window_size, 24)))
    X_test = numpy.asarray(numpy.reshape(X_test, (X_test.shape[0], window_size, 24)))
    y_train = numpy.asarray(y_train)
    y_test = numpy.asarray(y_test)

    numpy.random.seed(0)
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True, dropout=0.15, recurrent_dropout=0.1))
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.1,return_sequences=True, recurrent_dropout=0.1))
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(24))
    optimizer = keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.load_weights('static/model_weights/best_RNN_weights.hdf5')
    app.run()
