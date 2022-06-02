import tensorflow as tf

#MNISTデータセットをロード
mnist = tf.keras.datasets.mnist

#サンプルを整数型→浮動小数点数に変換
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#層を積み重ねてモデルを構築
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

#モデルをベースにクラスごとにロジット/対数オッズ比のスコアを算出
predictions = model(x_train[:1]).numpy()

#クラス毎にロジットの確率へ変換
tf.nn.softmax(predictions).numpy()

#標本についてクラスごとに損失のスカラーを返す
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#クラスの正しい確率の対数をとって符号を反転
loss_fn(y_train[:1], predictions).numpy()

#未訓練モデルはランダムに近い確率を出力
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#損失を最小化するようにモデルのパラメータ調整
model.fit(x_train, y_train, epochs=5)

#モデル性能を検査
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])