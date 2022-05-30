import numpy as np
import pandas as pd
import os
import shutil


def contruir_estrutura(nova_pasta_origem, origem):

    if os.path.exists("dataset/dog-breed-identification") == True:
        origem = os.path.join("dataset/dog-breed-identification",origem)
        etiquetas_csv = pd.read_csv("dataset/dog-breed-identification/labels.csv")
    else:
        origem = os.path.join("dataset",origem)
        etiquetas_csv = pd.read_csv("dataset/labels.csv")
    caminho_dt_treino = origem

    nome_arquivos = [caminho_dt_treino + nome_arquivo + ".jpg" for nome_arquivo in etiquetas_csv["id"]]
    etiquetas = etiquetas_csv["breed"].to_numpy() # convert labels column to NumPy array

    # cria a pasta para o novo dataset organizado por raças
    if os.path.exists(nova_pasta_origem) == False:
        os.mkdir(nova_pasta_origem)

    # cria as pastas de cada raca
    for raca_cachorro in np.unique(etiquetas):
        nova_pasta = raca_cachorro
        endereco_novo = os.path.join(nova_pasta_origem, nova_pasta)
        if os.path.exists(endereco_novo) == False:
            os.mkdir(endereco_novo)


    # move as imagens para sua determinada pasta
    for index in range(len(nome_arquivos)):
        endereco_imagem = nome_arquivos[index]
        nome_raca = etiquetas[index]
        src_path = endereco_imagem
        dst_path = os.path.join(nova_pasta_origem, nome_raca)
        shutil.move(src_path, dst_path)

