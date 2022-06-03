from load_training_data import load_df

import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import timedelta
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.DataFrame()



def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))



def normalize(data):
    norm_arr = scaler.fit_transform(data)
    return pd.DataFrame(norm_arr)



def inverse_normalize(data):
    return scaler.inverse_transform(data)



def generate_train_data(df):
    x_train, y_train = [], []
    for i in range(df['date'].shape[0] - 45):
        y_train.append( np.array(df['operating_reserve(norm)'].iloc[i+30:i+45]) )
        x_train.append( np.array(df['operating_reserve(norm)'].iloc[i:i+30]) )
    return np.array(x_train), np.array(y_train)



def train():
    from keras.layers import Dense, Dropout, Flatten, LSTM, TimeDistributed
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(LSTM(256, input_shape=(30, 1), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(30, activation='linear'))
    model.add(Dense(15, activation='linear'))
    model.compile(optimizer='rmsprop', loss=root_mean_squared_error, metrics=['accuracy'])
    model.summary()

    x_train, y_train = generate_train_data(df)
    callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
    history = model.fit(x_train, y_train, epochs=300, batch_size=5, validation_split=0.1, callbacks=[callback], shuffle=True)

    '''pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()'''

    model.save('lstm.h5')



def pred(output_file_name):
    from keras.models import load_model

    model = load_model('lstm.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error})

    df_row_count = df['date'].shape[0]
    df_test = normalize( np.array(df['operating_reserve(MW)'].iloc[df_row_count-30:df_row_count]).reshape(-1, 1) )
    y_pred = inverse_normalize( model.predict(np.array(df_test).reshape((1,30,1))) )
    
    submission = pd.DataFrame({
        'date': df['date'].iloc[df_row_count-15:df_row_count].apply( lambda x: (x+timedelta(days=15)).strftime("%Y%m%d") ),
        'operating_reserve(MW)': np.array(np.round(y_pred[0]), dtype=int)
    })
    submission.to_csv(output_file_name, index=False)



# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    parser.add_argument('--train',
                        default=False,
                        help='whether to train')
    args = parser.parse_args()

    # You can modify the following part at will.

    if os.path.exists(args.training) == True:
        df = pd.read_csv(args.training)
    else:
        df = load_df()
    df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y-%m-%d')
    df.rename(columns={'日期': 'date', '備轉容量(MW)': 'operating_reserve(MW)'}, inplace=True)
    df['operating_reserve(norm)'] = normalize( np.array(df['operating_reserve(MW)']).reshape(-1, 1) )
    
    if args.train == 'True':
        train()
        pred(args.output)
    else:
        pred(args.output)