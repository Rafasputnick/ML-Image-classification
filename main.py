import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from genericpath import exists, getsize
from keras import layers
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          Rescaling, ZeroPadding2D)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from organizador import contruir_estrutura, ler_racas

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hiperparametros
image_width = 100
image_height = 100
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

batch_size = 32
# numero de passos para o treinamento da ML terminar
# https://radiopaedia.org/articles/epoch-machine-learning#:~:text=An%20epoch%20is%20a%20term,of%20data%20is%20very%20large).
epochs = 1
learning_rate = 1e-2

if (exists("dt_treino") == False) or getsize("dt_treino") == 0 or getsize("dt_treino/affenpinscher") == 0:
    contruir_estrutura()

racas = ler_racas()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dt_treino",
    # validation_split=0.2,
    # subset="training",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
    labels='inferred',
    class_names=racas
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dt_validacao",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
    labels='inferred',
    class_names=racas
)

# aumenta a variabilidade do dataset fazendo algumas alterações aleatorias
aumentar_dataset = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.5),
        layers.RandomZoom(0.5),
    ]
)

# plota um grafico para ver as alterações em 1 imagem
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = aumentar_dataset(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

# plt.show()


# normaliza a imagem de 255,255,255 para escala de 0 à 1
augmented_train_ds = train_ds.map(
   lambda x, y: (aumentar_dataset(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def make_model(input_shape):
    with tf.device('/CPU:0'):
        inputs = keras.Input(shape=input_shape)
        x = aumentar_dataset(inputs)

        x = Conv2D(32, 5, activation='relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(64, 4, activation='relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, 3, activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)

        outputs = layers.Dense(120, activation="softmax")(x)
        return keras.Model(inputs, outputs)


model = make_model(image_shape)



# with tf.device('/CPU:0'):
#     model = tf.keras.models.Sequential([
#             tf.keras.layers.experimental.preprocessing.Rescaling(
#                 1./ 255.0,
#                 input_shape = image_size + (3,)
#             ),
#             aumentar_dataset,
#             tf.keras.layers.Conv2D(16, 3, padding = 'same', activation='relu'),
#             tf.keras.layers.MaxPooling2D(),
            
#             tf.keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
#             tf.keras.layers.MaxPooling2D(),
            
#             tf.keras.layers.Conv2D(64, 3, padding = 'same', activation='relu'),
#             tf.keras.layers.MaxPooling2D(),
            
#             tf.keras.layers.Flatten(),            

#             # tf.keras.layers.Dense(20, activation = 'relu'),

#             tf.keras.layers.Dropout(0.1),
#             tf.keras.layers.Dense(1, activation = 'sigmoid')
#         ])

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3),
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Precisa instalar o Graphviz, para rodar a linha abaixo
# só necessário rodar quando alterar o modelo
# keras.utils.plot_model(model, show_shapes=True)

historico = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

#Plota os graficos sobre o fit
# plt.figure(0)
# plt.plot(historico.history['accuracy'], label='training accuracy')
# plt.plot(historico.history['val_accuracy'], label='val accuracy')
# plt.title('Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()
# plt.figure(1)
# plt.plot(historico.history['loss'], label='training loss')
# plt.plot(historico.history['val_loss'], label='val loss')
# plt.title('Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# Segunda maneira
accuracy = historico.history['accuracy']
val_accuracy = historico.history['val_accuracy']
loss = historico.history['loss']
val_loss = historico.history['val_loss']
epochs_range = range(epochs)

plt.gcf().clear()
plt.figure(figsize = (15, 8))

plt.subplot(1, 2, 1)
plt.title('Training and Validation Accuracy')
plt.plot(epochs_range, accuracy, label = 'Training Accuracy')
plt.plot(epochs_range, val_accuracy, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')

plt.subplot(1, 2, 2)
plt.title('Training and Validation Loss')
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'lower right')

plt.show()


dataset_test_loss, dataset_test_accuracy = model.evaluate(val_ds)

print('Dataset Test Loss:     %s' % dataset_test_loss)
print('Dataset Test Accuracy: %s' % dataset_test_accuracy)


def plot_dataset_predictions(dataset):

    features, labels = val_ds.as_numpy_iterator().next()

    predictions = model.predict_on_batch(features).flatten()
    # predictions = tf.where(predictions < 0.5, 0, 1)

    print('Labels:      %s' % labels[0])
    # print('Index:      %s' % labels[0].index(1))
    print('Predictions: %s' % predictions.numpy())

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for i in range(9):

        plt.subplot(3, 3, i + 1)
        plt.axis('off')

        plt.imshow(features[i].astype('uint8'))
        plt.title(racas[predictions[i]])

plot_dataset_predictions(train_ds)

model.save("model_folder")

modelo_salvo = tf.keras.models.load_model('model_folder')

def predict(image_file):

    image = tf.keras.preprocessing.image.load_img(image_file, target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)

    # prediction = model.predict(image)[0][0]
    prediction = model.predict(image)
    print()


    # print('Prediction: {0} | {1}'.format(prediction, ('cat' if prediction < 0.5 else 'dog')))

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

predict('dataset/test/0a4ef19459cd2100977b052de5f46231.jpg')
predict_url('Imagem_selecionada', 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Black-Magic-Big-Boy.jpg')
predict_ramdom()
