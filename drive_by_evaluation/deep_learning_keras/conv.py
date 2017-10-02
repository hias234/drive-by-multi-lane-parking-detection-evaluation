from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Conv1D, GlobalMaxPool1D, GlobalAveragePooling1D


def create_conv_model(dataset, x_train, y_train):
    maxlen = 1026
    embedding_dims = 50
    max_features = 500

    #x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    #print('x_train[0]', x_train.shape)
    #print('y_train[0]', y_train.shape)

    # x_train = np.random.random((100, 100, 3))
    # y_train = keras.utils.to_categorical(np.random.randint(4, size=(100, 1)), num_classes=len(dataset.class_labels))
    # x_test = np.random.random((20, 100, 3))
    # y_test = keras.utils.to_categorical(np.random.randint(4, size=(20, 1)), num_classes=len(dataset.class_labels))
    # print('x_train[0]', x_train[0])
    # print('y_train[0]', y_train[0])

    filters = 64
    kernel_size = 5
    hidden_dims = 50

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Embedding(max_features,
                        embedding_dims,
                        #input_length=maxlen
                        ))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    # model.add(Conv1D(filters,
    #                 kernel_size,
    #                 padding='valid',
    #                 activation='relu',
    #                 strides=1))
    # we use max pooling:
    # model.add(GlobalMaxPooling1D())

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    # model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(64, 5, activation='relu', input_shape=(len(x_train[0]),1024)))
    # model.add(Conv1D(64, 5, activation='relu'))
    # model.add(MaxPooling1D(5))
    # model.add(Conv1D(128, 3, activation='relu'))
    # model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Dense(len(dataset.class_labels), activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)

    return model