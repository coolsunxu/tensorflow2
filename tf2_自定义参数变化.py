
from tensorflow import keras

starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5,
    cycle=True)

a = learning_rate_fn(10000) # 这里使用的是回调函数 __call__

print(a)


