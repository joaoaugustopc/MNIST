import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random


fashion_mnist = keras.datasets.fashion_mnist

(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

print("train shape: ", Xtrain.shape)
print("test shape: ", Xtest.shape)

print("classe:", Ytrain)

contagem = np.bincount(Ytrain)
print("Contagem de classes: ", contagem)



class_names = ['Camisetas/Top', 'Calça', 'Suéter', 'Vestidos', 'Casaco',
               'Sandálias', 'Camisas', 'Tênis', 'Bolsa', 'Botas']


Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Xtrain[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[Ytrain[i]])
plt.show()

"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=10)

model.save('fashion_mnist.h5')
"""

model = keras.models.load_model('fashion_mnist.h5')

test_loss, test_acc = model.evaluate(Xtest, Ytest, verbose=2)

print('\nTest accuracy:', test_acc)

posAleatoria = random.randint(0, len(Xtest) - 1)

imagem = Xtest[posAleatoria, :, :]

plt.imshow(imagem, cmap=plt.cm.binary)
plt.show()

prediction = model.predict(imagem.reshape(1, 28, 28))

print("Predição: ", class_names[np.argmax(prediction)])
print("Real: ", class_names[Ytest[posAleatoria]])



