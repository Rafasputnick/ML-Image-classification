import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
from tensorflow import keras
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.models import Sequential

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = keras.utils.to_categorical(trainY)
	testY = keras.utils.to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(keras.layers.Flatten(input_shape=[28, 28]))
	model.add(keras.layers.Dense(300, activation="relu"))
	model.add(keras.layers.Dense(100, activation="relu"))
	model.add(keras.layers.Dense(10, activation="softmax"))
	model.summary()
	return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	
	model.compile(
		 optimizer=keras.optimizers.RMSprop(),  # Optimizer
		# Loss function to minimize
		loss=keras.losses.SparseCategoricalCrossentropy(),
		# List of metrics to monitor
		metrics=[keras.metrics.SparseCategoricalAccuracy()],
	) 
	
	print("Fit model on training data")
	history = model.fit(
		trainX,
		trainY,
		batch_size=64,
		epochs=2,
		# We pass some validation for
		# monitoring validation loss and metrics
		# at the end of each epoch
		validation_data=(testX, testY),
	)

	# # fit model
	# history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)

	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)



# run_test_harness()


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = keras.utils.get_file(
    origin=dataset_url, fname="flower_photos", untar=True
)

dataset = keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(180, 180), batch_size=64
)

print(type(dataset))



# load dataset
# (trainX, trainy), (testX, testy) = cifar10.load_data()

# trainY =keras.utils.to_categorical(trainY)
# testY = keras.utils.to_categorical(testY)

# trainX, trainY, testX, testY = load_dataset()

# trainX, testX = prep_pixels(trainX, testX)
# trainY, testY = prep_pixels(trainY, testY)

# define_model()

# # summarize loaded dataset
# print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))

# print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
# # plot first few images
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	pyplot.imshow(trainX[i])
# # show the figure
# pyplot.show()