import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

mnist = tf.keras.datasets.mnist



def visualizaNumero(classe, X, y):
    qtImagens = 8
    print ("Classe selecionada: ", classe)
    XSelecionados = X[y == classe]
    fig, axs = plt.subplots(1, qtImagens, figsize=(qtImagens, 1))
    for c in range(qtImagens):
        posAleatoria = random.randint(0, len(XSelecionados) - 1)
        axs[c].imshow(XSelecionados[posAleatoria,:,:], cmap = plt.get_cmap('gray'))
        axs[c].axis('off')


def main():
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

    print("train shape: ", Xtrain.shape)
    print("test shape: ", Xtest.shape)
    """train shape:  (60000, 28, 28)
    test shape:  (10000, 28, 28)"""

    contagem = np.bincount(Ytrain)
    print("Contagem de classes: ", contagem)
    #Contagem de classes:  [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]

    ytrainT = tf.keras.utils.to_categorical(Ytrain,10) # para cada valor de Ytrain, cria um vetor de 10 posições, com 1 na posição do valor e 0 nas outras
    ytestT = tf.keras.utils.to_categorical(Ytest,10) # para ter multiplas saidas no MLP com valores polarizados. Ou 0 ou 1 (10 classes)

    XtrainT = Xtrain/255.0 # para os valores ficarem entre 0 e 1
    XtestT = Xtest/255.0

    numPixels = XtrainT.shape[1] * XtrainT.shape[2]
    XtrainT = XtrainT.reshape(XtrainT.shape[0], numPixels) # transforma a matriz em um vetor para se adequar às entradas do MLP
    XtestT = XtestT.reshape(XtestT.shape[0], numPixels)

    numClasses = len(ytrainT[0]) # 10 classes

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(400, input_dim =numPixels, activation='relu'))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))


    #Visualização de números da base de dados MNIST
    """visualizaNumero(0, Xtrain, Ytrain)
    visualizaNumero(5, Xtrain, Ytrain)
    plt.show()
    """

if __name__ == "__main__":
    main()


