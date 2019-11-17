import paddle
import paddle.fluid as fluid

from models.VGG16 import *
from data import *
from config import *


def train():
    # 1.构建网络
    input = fluid.layers.data("input", shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data("label", shape=[1], dtype='int64')
    predict = VGG16(CLASS_NUM).net(input)

    cost = fluid.layers.mean(fluid.layers.softmax_with_cross_entropy(predict, label))
    acc = fluid.layers.accuracy(predict, label)

    opt = fluid.optimizer.Adam()
    opt.minimize(cost)

    # 2.准备训练
    init_program = fluid.default_startup_program()
    train_program = fluid.default_main_program()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    dataset = DataSet(DATA_PATH)
    train_reader = paddle.batch(dataset.train_data, batch_size=BATCH_SIZE)
    test_reader = paddle.batch(dataset.test_data, batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder([input, label], place)

    # 3.开始训练
    exe.run(init_program)
    for epoch in range(EPOCHS):
        for iter, data in enumerate(train_reader()):
            cost_value, acc_value = exe.run(train_program, feed=feeder.feed(data), fetch_list=[cost, acc])
            if iter % 4 ==0:
                print(f"Epoch {epoch}, Iter {iter}, Cost={cost_value[0]}, Acc={acc_value[0]}")


def main():
    train()


if __name__ == '__main__':
    main()
