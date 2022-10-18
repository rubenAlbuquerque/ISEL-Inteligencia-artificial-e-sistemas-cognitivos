import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras import activations
from keras import optimizers

# Variables
EPOCHS = 500  # 
HL_NEURONS = 2  # 
LearningRates = np.array([0.05, 0.25, 0.5, 1, 2])
LR = 0.03
MOMENTUM = 0.5  # Best - 0.5
SHUFFLE = False  # 
LOSS_TARGET = 0.1

# Data
training_data = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
target_data   = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)

loss_target = np.array([LOSS_TARGET for i in range(EPOCHS)])

# Model
def modelo(LR, MOMENTUM, SHUFFLE, BINARY = True):
    model = Sequential()
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(2, activation='softmax'))


    model.compile(  loss=tf.keras.metrics.categorical_crossentropy, 
                    optimizer=optimizers.SGD(
                        learning_rate=LR,
                        momentum=MOMENTUM),
                    metrics=['accuracy'])

    history = model.fit(training_data,
                        target_data,
                        # shuffle=SHUFFLE,
                        epochs=EPOCHS,
                        verbose=0) #, use_multiprocessing=True)
    return model, history


for lr in range(len(LearningRates)):
    fig, plots = plt.subplots(2, 5)
    fig.suptitle(f'LR = {LearningRates[lr]} | MOMENTUM = {MOMENTUM}')

    for i in range(2):
        for j in range(5):
            model, history = modelo(LR, MOMENTUM, SHUFFLE, BINARY = True)
            plots[i, j].plot(np.array(history.history['loss']), 'o')
            plots[i, j].plot(loss_target, 'o')


print("\n\nresukltado:")
pred = model.predict(training_data)
for name, valor in zip('vdd, predict'.split(','),[target_data, pred]):
    print()
    print(name)
    print(valor)