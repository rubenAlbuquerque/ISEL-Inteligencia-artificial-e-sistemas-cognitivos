import time
import math
import numpy as np
import random as rnd
from matplotlib import pyplot
from keras.layers.core import Dense
from keras import activations
from keras import optimizers
from keras import models

MATRIX_SIZE = 4

SAMPLES = 1000  # 2**(MATRIX_SIZE**2-1)
EPOCHS = 10

TRAIN_TEST_RATIO = 0.8

LOAD_MODEL = True
PATH = 'Objetivo 1/models/2/'
FOLDER = '1_4x4_s10k_e1k_1024x3'

sample = np.array([[1, 1, 0, 1],
                   [1, 0, 1, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1]])


#
#
#


def get_data():
    """ Gerar array de arrays únicos de tamanho MATRIX_SIZE**2

    Returns:
        NumPy Array: Array de arrays únicos de tamanho MATRIX_SIZE**2
    """

    data = np.array([], dtype=int)
    numbers = np.array([], dtype=int)

    while len(numbers) < SAMPLES:
        n = rnd.randint(0, 2 ** (MATRIX_SIZE ** 2) - 1)

        if n in numbers:
            if len(numbers) == SAMPLES:
                break
            continue

        numbers = np.append(numbers, n)

        matrix = bin(n).removeprefix('0b').rjust(MATRIX_SIZE ** 2, '0')
        matrix = np.array([int(i) for i in matrix])
        data = np.append(data, matrix)

    data = np.reshape(data, (SAMPLES, MATRIX_SIZE ** 2))

    return data


def get_matrix_outputs(matrix):
    """ Gerar array com os números associados das linhas e colunas da matriz

    Args:
        matrix (NumPy Array): Matriz quadrada

    Returns:
        NumPy Array: [linha1, ..., linhaN, coluna1, ..., colunaN]
    """

    size = int(math.sqrt(np.size(matrix)))

    matrix = np.reshape(matrix, (size, size))

    outputs = np.zeros((2, size), dtype=np.int)

    for row in range(size):
        space = True
        for n in matrix[row, :]:
            if n == 0:
                space = True
            elif space:
                outputs[0][row] *= 10
                space = False
            outputs[0][row] += n

    for col in range(size):
        space = True
        for n in matrix[:, col]:
            if n == 0:
                space = True
            elif space:
                outputs[1][col] *= 10
                space = False
            outputs[1][col] += n

    return np.reshape(outputs, size * 2)


#
#
#


def get_model():
    """ Get model

    Returns:
        Sequential Model: model with MATRIX_SIZE*2 inputs and MATRIX_SIZE**2 outputs
    """

    model = models.Sequential()
    model.add(Dense(2 ** 10, input_dim=MATRIX_SIZE * 2, activation=activations.relu))
    model.add(Dense(2 ** 10, activation=activations.relu))
    model.add(Dense(2 ** 10, activation=activations.relu))
    model.add(Dense(MATRIX_SIZE ** 2, activation=activations.sigmoid))

    model.compile(optimizer=optimizers.adam_v2.Adam(), loss='mean_squared_error', metrics=["accuracy"])

    model.summary()

    return model


#
#
#


def __main__():
    # Generate data
    print('[DEBUG] Generating target data...')
    data = get_data()

    # Test loaded model
    if LOAD_MODEL:
        print('[DEBUG] Loding model...')
        model = models.load_model(PATH + FOLDER)

    # Train and save model
    else:
        target_data = data[:int(SAMPLES * TRAIN_TEST_RATIO)]
        # print('Target Data\n', target_data)

        print('[DEBUG] Generating train data...')
        train_data = np.array([get_matrix_outputs(m) for m in target_data], dtype=int)
        # print('Train Data\n', train_data)

        # Train model
        model = get_model()

        print('[DEBUG] Training model...')
        start = time.time()
        history = model.fit(train_data, target_data,
                            epochs=EPOCHS,
                            verbose='auto',
                            validation_split=0.2,
                            validation_freq=1,
                            use_multiprocessing=True)

        model.summary()

        print(f'[DEBUG] Took {time.time() - start}s | Samples: {SAMPLES} | Epochs: {EPOCHS}')

        # Save model
        model.save(PATH + 'last', save_format='tf')

        # Plot performance
        pyplot.plot(history.history['loss'], label='loss')
        pyplot.plot(history.history['accuracy'], label='accuracy')
        pyplot.plot(history.history['val_loss'], label='val_loss')
        pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
        pyplot.legend()
        pyplot.savefig(PATH + 'last.png')
        pyplot.show()

    # Test model
    test_target_data = data[int(SAMPLES * TRAIN_TEST_RATIO):]
    test_data = np.array([get_matrix_outputs(m) for m in test_target_data], dtype=int)

    print('[DEBUG] Evaluating model...')
    print('\n[Loss, Accuracy]', model.evaluate(test_data, test_target_data))

    sample_input = get_matrix_outputs(np.reshape(sample, MATRIX_SIZE ** 2))
    result = np.array(np.reshape(
        model.predict(np.array([sample_input])).round(),
        (MATRIX_SIZE, MATRIX_SIZE)
    ), dtype=int)

    print('\nInput\n', sample_input)
    print('Expected\n', sample)
    print('Output\n', result)
    print('Assert:', np.array_equal(result, sample))


print('\n' + '-'*100 + '\n')
__main__()
