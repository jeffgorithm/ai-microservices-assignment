import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import pickle

def create_model(num_layers, num_neurons, n_cols):
    model = Sequential()

    #Input layer
    model.add(Dense(num_neurons, activation='relu', input_shape=(n_cols,)))
    
    #Add hidden layers
    for _ in range(0, num_layers):
        model.add(Dense(num_neurons, activation = 'relu'))
    
    #Add output layer
    model.add(Dense(1))

    return model

train_df = pd.read_csv("data.csv")
train_df = train_df.drop(columns = ['current_time', 'hour', 'minute', 'round'])
train_df.reset_index()

#Extract Features and Label Columns
train_X = train_df.drop(columns=['duration'])
train_Y = train_df[['duration']]

#Label Encoding
encoder = LabelEncoder()
encoder.fit(train_X['bus_stop_code'])
encoded_bus_stops = encoder.transform(train_X['bus_stop_code'])

#Save encoder
pickle.dump(encoder, open("input_encoder.pkl", 'wb'))

#Get numerical values as float from input data
time_X = train_X['time'].values
time_X = time_X.astype('float32')

train_Y = train_Y.values
train_Y = train_Y.astype('float32')

#Scale numerical data (time and duration)
input_scalar, output_scalar = MinMaxScaler(), MinMaxScaler()
input_scalar.fit(time_X.reshape(-1, 1))
time_X = input_scalar.transform(time_X.reshape(-1, 1))
output_scalar.fit(train_Y)
train_Y = output_scalar.transform(train_Y)

#Concatenate scaled values with time values
train_X = np.stack(
    (encoded_bus_stops, train_X['day'].values, np.squeeze(time_X)),
    axis=1
    )

#Save Scalar Files
pickle.dump(input_scalar, open("input_scalar.pkl", 'wb'))
pickle.dump(output_scalar, open("output_scalar.pkl", 'wb'))

train_X, test_X, train_Y, test_Y = train_test_split(
    train_X, 
    train_Y, test_size=0.2,
    shuffle=True,
    random_state=88
    )

#get number of columns in training data
n_cols = train_X.shape[1]

num_layers = 10
num_neurons = 20

#Create and conpile model with Adam optimizer, L2/MSE for loss function
model = create_model(num_layers, num_neurons, n_cols)
model.compile(
    optimizer='adam', 
    loss='mean_squared_error', 
    metrics = ['mae']
    )

#Save model at checkpoint
checkpoint = ModelCheckpoint(
    "model.h5", 
    monitor='loss', 
    verbose=0, 
    save_best_only=True, 
    mode='min'
    )

#Train Model
model.fit(
    train_X,
    train_Y, 
    epochs=50,
    batch_size = 32, 
    verbose = 1, 
    validation_data=(test_X, test_Y),
    callbacks = checkpoint
    )

#Evaluate Model
train_score = model.evaluate(train_X, train_Y, verbose = 0)
test_score = model.evaluate(test_X, test_Y, verbose=0)
