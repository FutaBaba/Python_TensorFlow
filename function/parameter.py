from tabnanny import verbose
import tensorflow as tf

#手書きの数字の画像
mnist = tf.keras.datasets.mnist

#x_trainは訓練用の画像データ、y_trainは訓練用の数字のラベルの配列
#x_testはテスト用の画像データ、y_testはテスト用の数字のラベルの配列
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#モデル
def make_model(n):
    model = tf.keras.models.Sequential([
    #入力層
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #ノード数がn、活性化関数がreluである層
    tf.keras.layers.Dense(n, activation='relu'),
    #0.2の割合で出力を0にする
    tf.keras.layers.Dropout(0.3),
    #出力層
    tf.keras.layers.Dense(10)
    ])
    return model

def parameter_test(model,n):
    accuracy = 0
    parameter_number = n
    for i in range(0,100):
        mymodel = model(n)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mymodel.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
        mymodel.fit(x_train, y_train, epochs=5)
        model_accuracy = mymodel.evaluate(x_test,  y_test, verbose=2)[1]
        print("babalog,",n,",",model_accuracy)
        if  model_accuracy > accuracy:
            accuracy = model_accuracy
            parameter_number = n
        n = n+10
    print(parameter_number)

parameter_test(make_model, 50)