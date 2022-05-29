import numpy as np
import pandas as pd
import os
import shutil


etiquetas_csv = pd.read_csv("dataset/labels.csv")
caminho_dt_treino = "dataset/train/"

nome_arquivos = [caminho_dt_treino + nome_arquivo + ".jpg" for nome_arquivo in etiquetas_csv["id"]]
etiquetas = etiquetas_csv["breed"].to_numpy() # convert labels column to NumPy array

# cria a pasta para o novo dataset organizado por raças
os.mkdir("new_dataset")

# cria as pastas de cada raca
for raca_cachorro in np.unique(etiquetas):
    nova_pasta = raca_cachorro
    endereco_pasta = "new_dataset"
    endereco_novo = os.path.join(endereco_pasta, nova_pasta)
    os.mkdir(endereco_novo)


# move as imagens para sua determinada pasta
for index in range(len(nome_arquivos)):
    endereco_imagem = nome_arquivos[index]
    nome_raca = etiquetas[index]
    src_path = endereco_imagem
    dst_path = os.path.join("new_dataset", nome_raca)
    shutil.move(src_path, dst_path)

