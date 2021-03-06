import math
import os
import shutil
from tkinter.messagebox import NO

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def carregar_dataset():
    if os.path.exists("dataset/dog-breed-identification"):
        origem = "dataset/dog-breed-identification/"
        etiquetas_csv = pd.read_csv(origem + "labels.csv")
    else:
        origem = "dataset/"
        etiquetas_csv = pd.read_csv( origem + "labels.csv")
    
    return origem, etiquetas_csv

def ler_racas():
    origem, etiquetas_csv = carregar_dataset()
    etiquetas = etiquetas_csv["breed"].to_numpy()
    # return np.unique(etiquetas).ravel()
    return list(np.unique(etiquetas))


def contruir_estrutura(racas = None):
    origem, etiquetas_csv = carregar_dataset()
    
    etiquetas = etiquetas_csv["breed"].to_numpy()

    if racas is None:
        racas = np.unique(etiquetas)
    
    cachorro_raca_geral = {}
    for raca in racas:
        cachorro_raca_geral[raca] = etiquetas_csv.loc[etiquetas_csv['breed'] == raca]['id'].to_numpy()
    
    if not os.path.exists("class_dataset"):
        os.mkdir("class_dataset")
    
    # cria as pastas de cada raca
    for raca in racas:
        pasta_raca = os.path.join("class_dataset", raca)

        if not os.path.exists(pasta_raca):
            os.mkdir(pasta_raca)

        for id_imagem in cachorro_raca_geral[raca]:
            endereco_imagem = f"{origem}train/{id_imagem}.jpg"
            pasta_raca = f"class_dataset/{raca}"
            shutil.move(endereco_imagem, pasta_raca)

    # cachorro_raca_treino = {}
    # cachorro_raca_validacao = {}

    # for raca in racas:
    #     quantidade_total = len(cachorro_raca_geral[raca])
    #     quantidade_treino = math.floor(quantidade_total * 0.8)

    #     cachorro_raca_treino[raca] = cachorro_raca_geral[raca][0:quantidade_treino]
    #     cachorro_raca_validacao[raca] = cachorro_raca_geral[raca][quantidade_treino:]

    # # cria a pasta dt_treino para o novo dataset organizado por raças
    # if not os.path.exists("dt_treino"):
    #     os.mkdir("dt_treino")

    # # cria a pasta dt_validacao para o novo dataset organizado por raças
    # if not os.path.exists("dt_validacao"):
    #     os.mkdir("dt_validacao")

    # # cria as pastas de cada raca
    # for raca in racas:
    #     pasta_raca_treino = os.path.join("dt_treino", raca)
    #     pasta_raca_validacao = os.path.join("dt_validacao", raca)

    #     if not os.path.exists(pasta_raca_treino):
    #         os.mkdir(pasta_raca_treino)
        
    #     if not os.path.exists(pasta_raca_validacao):
    #         os.mkdir(pasta_raca_validacao)

    # for raca in racas:
    #     for id_imagem in cachorro_raca_treino[raca]:
    #         endereco_imagem = f"{origem}train/{id_imagem}.jpg"
    #         pasta_raca = f"dt_treino/{raca}"
    #         shutil.move(endereco_imagem, pasta_raca)
        
    #     for id_imagem in cachorro_raca_validacao[raca]:
    #         endereco_imagem = f"{origem}train/{id_imagem}.jpg"
    #         pasta_raca = f"dt_validacao/{raca}"
    #         shutil.move(endereco_imagem, pasta_raca)


def converter_imagens(diretorios):
    for diretorio in diretorios:
        diretorio_rel = os.path.join('class_dataset', diretorio)

        arquivos = next(os.walk(diretorio_rel), (None, None, []))[2]
        for arquivo in arquivos:
            nome_diretorio = os.path.join(diretorio_rel, arquivo)
            imagem = cv2.imread(nome_diretorio)
            imagem = tratar_imagem(imagem)
            cv2.imwrite(nome_diretorio, imagem)

def converter_imagem(diretorio, dim):
    imagem = cv2.imread(diretorio)
    imagem = tratar_imagem(imagem)
    cv2.imwrite("imagem_trata.jpg", imagem)

    #imagem = cv2.resize(imagem, dim, cv2.INTER_NEAREST)
    imagem = tf.keras.preprocessing.image.load_img("imagem_trata.jpg", target_size = dim, color_mode='grayscale')
    
    return imagem

def tratar_imagem(imagem):

    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_blur = cv2.GaussianBlur(imagem_gray, (5,5), 0)

    sobelxy = cv2.Sobel(src=imagem_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    return sobelxy
