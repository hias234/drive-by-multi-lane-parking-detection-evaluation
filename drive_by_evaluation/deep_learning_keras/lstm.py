from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense


def create_lstm_model(dataset, x_train, y_train):
    max_features = 128
    hidden_dims = 64

    model = Sequential()
    model.add(Embedding(max_features, output_dim=256))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=3,
              batch_size=128)

    return model