import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dropout,
    Dense
)


def build_model(input_size: int, output_size: int) -> tensorflow.keras.models.Sequential:
    model = Sequential()
    model.add(LSTM(700, input_shape=(input_size, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(700))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='softmax'))
    return model
