import ctypes
from tensorflow import keras

epoch_atual = 0
class CustomCallback(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        global epoch_atual 
        epoch_atual = epoch
