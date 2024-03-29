from tabnanny import verbose
import tensorflow as tf

#手書きの数字の画像
mnist = tf.keras.datasets.mnist

#x_trainは訓練用の画像データ、y_trainは訓練用の数字のラベルの配列
#x_testはテスト用の画像データ、y_testはテスト用の数字のラベルの配列
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#モデル
model = tf.keras.models.Sequential([
  #入力層
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #ノード数が128、活性化関数がreluである層
  tf.keras.layers.Dense(117, activation='relu'),
  #0.2の割合で出力を0にする
  tf.keras.layers.Dropout(0.3),
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