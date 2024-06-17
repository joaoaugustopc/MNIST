import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    mnist = tf.keras.datasets.mnist
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
    print("train shape: ", Xtrain.shape)
    print("test shape: ", Xtest.shape)
    
    #escalonando as intensidades dos pixels para o intervalo [0,1]
    x_valid, x_train = Xtrain[:5000] / 255.0, Xtrain[5000:] / 255.0
    y_valid, y_train = Ytrain[:5000], Ytrain[5000:]
    
    model = keras.models.Sequential()
    #convertendo imagem para entrada de matriz 1D
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    #add saída e função softmax para classificação multiclasse
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    #sparse_categorical_crossentropy: labels são inteiros entre 0 e 9 e classes exclusivas
    #sgd: gradiente descendente estocástico -> algoritmo de retropropagação
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    
    history = model.fit(x_train, y_train, epochs=30,
                        validation_data=(x_valid, y_valid))
    
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
    X_new = Xtest[:3]
    y_proba = model.predict(X_new)
    print(y_proba.round(2))

if __name__ == "__main__":
    main()