from tabnanny import verbose
import tensorflow as tf

#手書きの数字の画像
mnist = tf.keras.datasets.mnist

#x_trainは訓練用の画像データ、y_trainは訓練用の数字のラベルの配列
#x_testはテスト用の画像データ、y_testはテスト用の数字のラベルの配列
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#TensorFlow パラメータチューニング手動

#パラメータ付きモデルを受け取って最も良いパラメータを返す関数

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(128,(3,3),activation='relu', input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  #ノード数が128、活性化関数がreluである層
  tf.keras.layers.Dense(128, activation='relu'),
  #0.2の割合で出力を0にする
  tf.keras.layers.Dropout(0.2),
  #出力層
  tf.keras.layers.Dense(10)
])
model.summary()

#損失関数の設定
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
#ニューラルネットワークの訓練
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)