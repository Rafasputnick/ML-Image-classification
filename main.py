import os

import matplotlib.pyplot as plt
import tensorflow as tf
from genericpath import exists, getsize
from keras import layers
from tensorflow import keras

from organizador import contruir_estrutura

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if (exists("dt_treino") == False) or getsize("dt_treino") == 0 or getsize("dt_treino/affenpinscher") == 0:
    contruir_estrutura("dt_treino", "train/")

# if (exists("dt_validacao") == False) or getsize("dt_validacao") == 0 or getsize("dt_validacao/affenpinscher") == 0:
#     contruir_estrutura("dt_validacao", "test/")    

image_size = (224, 224)
# quantidade de exempplos de treino que serao usados em 1 iteraçao
# tipo "mini-batch mode" -> maior que um porem menor que o range do dataset usado
# geralmente é um numero que pode ser dividido pelo tamanho total do dataset
# https://radiopaedia.org/articles/batch-size-machine-learning?lang=gb
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dt_treino",
    validation_split=0.2,
    subset="training",
    #color_mode="grayscale",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dt_treino",
    validation_split=0.2,
    subset="validation",
    #color_mode="grayscale",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# plota um grafico para ver as alterações em 1 imagem
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

# plt.show()


# normaliza a imagem de 255,255,255 para escala de 0 à 1
augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))


train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


def make_model(input_shape, num_classes):
    
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

# numero de passos para o treinamento da ML terminar
# https://radiopaedia.org/articles/epoch-machine-learning#:~:text=An%20epoch%20is%20a%20term,of%20data%20is%20very%20large).
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]

res_str = "This image is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score)

print(res_str)

plt.figure(figsize=(5, 5))
# ax = plt.subplot(3)
plt.title("Imagem 6779")
plt.imshow(img)
# plt.text(0.5, 0, res_str , fontsize=20,
#      horizontalalignment='center')
plt.axis("off")

plt.show()


