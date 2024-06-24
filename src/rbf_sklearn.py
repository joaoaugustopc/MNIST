import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np


""" 
    Função que visualiza uma imagem da base de dados MNIST
    x: matrizes
    y: labels -> conteúdo da matriz 
    classe: numero a ser encontrado 

"""
def visualiza_numero(classe, x, y):
    qtd_img = 8  # quantidade de imagens a serem visualizadas
    print(f'Classe: {classe}')  # numero passado para encontrar a imagem
    x_selecionados = x[y == classe]  # selecionando as imagens da classe
    fig, axs = plt.subplots(nrows=1, ncols=qtd_img, figsize=(qtd_img, 1))  # criando a figura
    for col in range(qtd_img):
        rand_pos = random.randint(0, len(x_selecionados) - 1)  # selecionando uma imagem aleatória
        img_to_plot = x_selecionados[rand_pos, :].reshape(28, 28)  # Reshape the image to 2D
        axs[col].imshow(img_to_plot, cmap=plt.get_cmap("gray"))  # plotando a imagem
        axs[col].axis('off')  # removendo os eixos
    plt.show()
        
""" 
    Função para prever o número com modelo treinado
"""
def previsao_num(model, x_test):
    random_pos = random.randint(0, len(x_test) - 1)  # selecionando uma imagem aleatória
    img_to_pred = x_test[random_pos, :]  # selecionando a imagem
    img_to_pred_reshaped = img_to_pred.reshape(28, 28)  # reshape the image to 2D
    plt.imshow(img_to_pred_reshaped, cmap=plt.get_cmap('gray'))
    plt.show()

    img_to_pred = img_to_pred / 255.0  # normalizando a imagem
    img_to_pred = img_to_pred.reshape(1, 28 * 28)  # reshape

    # prediction
    prediction = model.predict_proba(img_to_pred)
    print(prediction)
    print("Prediction:", np.argmax(prediction))  # printando a previsão

def main():
    #carregando dados
    mnist = tf.keras.datasets.mnist
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
       
    #normalizando
    Xtrain_t = Xtrain/255.0
    Xtest_t = Xtest/255.0
    
    #redimensionando a imagem

    num_pixels = 28*28
    Xtrain_t = Xtrain_t.reshape(Xtrain.shape[0], num_pixels)
    Xtest_t = Xtest_t.reshape(Xtest.shape[0], num_pixels)
    
    # diminuindo o tamanho do dataset
    subset_size = 3000
    Xtrain_subset = Xtrain_t[:subset_size]
    Ytrain_subset = Ytrain[:subset_size]
        
        
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(Xtrain_subset, Ytrain_subset)
    
    visualiza_numero(3, Xtrain_subset, Ytrain_subset)
    previsao_num(gpc, Xtest_t)
    
if __name__ == "__main__":
    main()
    
    