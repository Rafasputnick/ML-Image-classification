import tensorflow as tf
from os.path import exists, getsize
from organizador import contruir_estrutura

if (exists("dt_treino") == False) or getsize("dt_treino") == 0 or getsize("dt_treino/affenpinscher") == 0:
    contruir_estrutura("dt_treino", "dataset/train/")

# if (exists("dt_validacao") == False) or getsize("dt_validacao") == 0 or getsize("dt_validacao/affenpinscher") == 0:
#     contruir_estrutura("dt_validacao", "dataset/test/")    

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
    color_mode="grayscale",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dt_treino",
    validation_split=0.2,
    subset="validation",
    color_mode="grayscale",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

print()


