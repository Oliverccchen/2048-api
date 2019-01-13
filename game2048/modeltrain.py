from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, BatchNormalization, Activation
from keras.models import Model, load_model
import keras
import numpy as np
import pandas as pd

filepath = '/home/ubuntu/cbq/traindata10.csv'
df = pd.read_csv(filepath)
cols = ["R1C1", "R1C2", "R1C3", "R1C4", \
        "R2C1", "R2C2", "R2C3", "R2C4", \
        "R3C1", "R3C2", "R3C3", "R3C4", \
        "R4C1", "R4C2", "R4C3", "R4C4"]
x = df[cols]
cols = ["direction"]
y = df[cols]
X = []
Y = []
action = [0]*4
number = len(y)
i = 1
while i < number:
    a = x.ix[i, :]
    A = np.array(a)
    b = np.reshape(A, (4, 4))
    c = b[:, :, np.newaxis]
    line = y.ix[i, 'direction']
    line = int(line)
    action[line] = 1
    X.append(c)
    Y.append(action)
    action = [0]*4
    i += 1
Y_TRAIN = np.array(Y)
X_TRAIN = np.array(X)


inputs = Input((4, 4, 1))
conv = inputs
FILTERS = 128
conv41 = Conv2D(filters=FILTERS, kernel_size=(4, 1), kernel_initializer='he_uniform')(conv)
conv14 = Conv2D(filters=FILTERS, kernel_size=(1, 4), kernel_initializer='he_uniform')(conv)
conv22 = Conv2D(filters=FILTERS, kernel_size=(2, 2), kernel_initializer="he_uniform")(conv)
conv33 = Conv2D(filters=FILTERS, kernel_size=(3, 3), kernel_initializer='he_uniform')(conv)
conv44 = Conv2D(filters=FILTERS, kernel_size=(4, 4), kernel_initializer='he_uniform')(conv)

hidden = concatenate([Flatten()(conv41), Flatten()(conv14), Flatten()(conv22), Flatten()(conv33), Flatten()(conv44)])
x = BatchNormalization()(hidden)
x = Activation('relu')(x)

for width in [512, 128]:
    x = Dense(width, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

outputs = Dense(4, activation='softmax')(x)


def main():
    model = load_model('/home/ubuntu/cbq/models9.h5')
    #model.summary()
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_TRAIN, Y_TRAIN, epochs=20)
    model.save('/home/ubuntu/cbq/models10.h5')


if __name__ == '__main__':
    main()






