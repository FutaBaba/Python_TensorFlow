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
  #入力
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #出力の次元数が128、activation関数がrelu(0以上のときだけ値をそのまま返す)
  tf.keras.layers.Dense(128, activation='relu'),
  #0.2の割合で入力を0にする
  tf.keras.layers.Dropout(0.2),
  #出力
  tf.keras.layers.Dense(10)
])

#[:1]は配列の最初の要素
predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)