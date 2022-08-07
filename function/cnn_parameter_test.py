import tensorflow as tf

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#reshape 元は(60000, 28, 28),(10000, 28, 28)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0

def set_parameter(model_function,layer_number,parameter_number):
    model = model_function(layer_number,parameter_number)
    params = model.count_params()
    if params < 100000:
        return set_parameter(model_function,layer_number,parameter_number + 1)
    else:
        model.summary()
        return model

def make_model(layer_number,parameter_number):
    model = models.Sequential()
    model.add(layers.Conv2D(parameter_number, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    for i in range(0,layer_number):
        model.add(layers.Conv2D(parameter_number, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(parameter_number, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(parameter_number, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def parameter_test(model_function,set_function):
    n = 0
    for i in range(0,3):
        mymodel = set_function(model_function,n,1)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mymodel.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
        mymodel.fit(train_images, train_labels, epochs=5)
        mymodel.evaluate(test_images,  test_labels, verbose=2)[1]
        n += 1

parameter_test(make_model,set_parameter)

def parameter_test_2(model_function):
    accuracy = 0
    parameter_number = 20
    for i in range(0,200):
        mymodel = model_function(1,n)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mymodel.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
        mymodel.fit(train_images, train_labels, epochs=5)
        model_accuracy = mymodel.evaluate(test_images,  test_labels, verbose=2)[1]
        if  model_accuracy > accuracy:
            accuracy = model_accuracy
            parameter_number = n
        n = n+10
    print(parameter_number)

#parameter_test_2(make_model)