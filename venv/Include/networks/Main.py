from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

plt.figure(figsize=(10, 10))

for i in range(100, 150):
    plt.subplot(5, 10, i - 100 + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])


x_train = x_train.reshape(60000, 784)

x_train = x_train / 255

print(y_train[0])

y_train = utils.to_categorical(y_train, 10)


print(y_train[0])

model = Sequential()

model.add(Dense(800, input_dim=784, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=200,
                    epochs=100,
                    verbose=1)


predictions = model.predict(x_train)

n = 0
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()


print(predictions[n])

np.argmax(predictions[n])


print(classes[np.argmax(predictions[n])])


np.argmax(y_train[n])

print(classes[np.argmax(y_train[n])])

print("succes end program")

