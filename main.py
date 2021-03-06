
import logging
from textwrap import indent
import utilidade
import json
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import custom_callback
from custom_callback import CustomCallback
from genericpath import exists, getsize
from keras import layers
from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                          Rescaling, GlobalAveragePooling2D)
from tensorflow import keras

from organizador import contruir_estrutura, converter_imagem, converter_imagens, ler_racas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 12335234

# Hiperparametros - 1
batch_size = 32
epochs = 10
learning_rate = 1e-4

# Fazer treinamento
train = False

# Imagem para testar
image_teste = 'validation/corgi2.jpg'

# Tratamento de falha

# diretorio do checkpoint
diretorio_checkpoint = "G:\checkpoint"

# Caso houver erro
error_recover = False
# Epoch do checkpoint
epoch_carregada = 0

# Hiperparametros - 2
image_width = 220
image_height = 220
image_color_channel = 1
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

racas = [
    "affenpinscher",
    "afghan_hound",
    "black-and-tan_coonhound",
    "blenheim_spaniel",
    "bloodhound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "cardigan"
]

if (train):

    if (exists("class_dataset") == False) or getsize("class_dataset") == 0 or getsize("class_dataset/affenpinscher") == 0:
        contruir_estrutura(racas)
        converter_imagens(racas)

    dataset_treino = tf.keras.preprocessing.image_dataset_from_directory(
        "class_dataset",
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='categorical',
        labels='inferred',
        class_names=racas,
        shuffle=True,
        subset='training',
        validation_split=0.2,
        seed=seed
    )

    dataset_validacao = tf.keras.preprocessing.image_dataset_from_directory(
        "class_dataset",
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='categorical',
        labels='inferred',
        class_names=racas,
        shuffle=True,
        subset='validation',
        validation_split=0.2,
        seed=seed
    )

    # aumenta a variabilidade do dataset fazendo algumas altera????es aleatorias
    aumentar_dataset = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ]
    )

    dataset_treino = dataset_treino.prefetch(buffer_size=32)
    dataset_validacao = dataset_validacao.prefetch(buffer_size=32)

    def make_model(input_shape):    
        model = tf.keras.models.Sequential([
            Rescaling(
                1./ 255.0,
                input_shape = input_shape
            ),
            aumentar_dataset,

            Conv2D(16, 3, activation='relu', padding='same'),
            MaxPooling2D(),
            Dropout(.2),
            Conv2D(32, 5, activation='relu'),
            MaxPooling2D(),
            Dropout(.2),
            Conv2D(64, 3, activation='relu', padding='same'),
            MaxPooling2D(),
            Dropout(.2),

            Flatten(),
            Dense(112, activation = 'relu'),
            Dropout(.3),

            Dense(10, activation = 'softmax')
        ])
        return model

    try:
        model = tf.keras.models.load_model('model_folder')
    except:
        model = make_model(image_shape)

    model.summary()

    callbacks = [
        #keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1),
        keras.callbacks.ModelCheckpoint(diretorio_checkpoint + "/cp_{epoch}.h5", mode="max", monitor="val_accuracy", save_best_only=True),
        CustomCallback()
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Precisa instalar o Graphviz, para rodar a linha abaixo
    # s?? necess??rio rodar quando alterar o modelo
    #keras.utils.plot_model(model, show_shapes=True)


    retry = True
    if error_recover:
        model.load_weights(diretorio_checkpoint + "/cp_" + epoch_carregada + ".h5")
        error = False
        
    historico = model.fit(dataset_treino, epochs=epochs, callbacks=callbacks, validation_data=dataset_validacao)
    model.save("model_folder")

    utilidade.plot_grafico(historico.history, "atual")
    historico_longo = json.load(open('train_history/history.json', 'r'))
    if len(historico_longo.keys()) > 0:
        for key in historico.history.keys():
            historico_longo[key].extend(historico.history[key])
    else:
        historico_longo = historico.history

    utilidade.plot_grafico(historico_longo, "longo")
    json.dump(historico_longo, open('train_history/history.json', 'w'), indent=4)

modelo_salvo = tf.keras.models.load_model('model_folder')
#salvar_h5_model = modelo_salvo.save("model_upload.h5")


def predict(imagem_file):
    print(imagem_file)
    image = converter_imagem(imagem_file, image_size)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)

    prediction = modelo_salvo.predict(image)
    for i in range(len(racas)):
        print(f"Prediction: {racas[i]:>30}| {prediction[0][i] * 100}")
    print(f"---------------------------------------------------------------")
    max_value_index = np.argmax(prediction[0])
    print(f"Prediction: {racas[max_value_index]}")

def predict_url(image_fname, image_origin):
    image_file = tf.keras.utils.get_file(image_fname, origin = image_origin)
    return predict(image_file)

def predict_ramdom():
    arquivos_test = os.listdir("dataset/test/")
    index_imagem = random.randrange(0, len(arquivos_test))

    nome_imagem = arquivos_test[index_imagem]
    image = tf.keras.preprocessing.image.load_img("dataset/test/" + nome_imagem, target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    
    # prediction = model.predict(image)[0][0]
    prediction = model.predict(image)
    print()

predict(image_teste)
# predict_url('Imagem_selecionada', 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Black-Magic-Big-Boy.jpg')
# predict_ramdom()
