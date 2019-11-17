from paddle.fluid.layers import fc, conv2d, pool2d, flatten
import paddle.fluid as fluid


class VGG16:
    def __init__(self, class_num):
        self.class_num = class_num

    def net(self, inputs):
        print(inputs.shape)

        x = conv2d(inputs, 64, 3, padding=1, act='relu')
        x = conv2d(x, 64, 3, padding=1, act='relu')
        x = pool2d(x, 2, pool_stride=2)
        print(x.shape)

        x = conv2d(x, 128, 3, padding=1, act='relu')
        x = conv2d(x, 128, 3, padding=1, act='relu')
        x = pool2d(x, 2, pool_stride=2)
        print(x.shape)

        x = conv2d(x, 256, 3, padding=1, act='relu')
        x = conv2d(x, 256, 3, padding=1, act='relu')
        x = conv2d(x, 256, 3, padding=1, act='relu')
        x = pool2d(x, 2, pool_stride=2)
        print(x.shape)

        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = pool2d(x, 2, pool_stride=2)
        print(x.shape)

        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = conv2d(x, 512, 3, padding=1, act='relu')
        x = pool2d(x, 2, pool_stride=2)
        print(x.shape)

        x = flatten(x)
        x = fc(x, 4096, act='relu')
        x = fc(x, 4096, act='relu')
        out = fc(x, self.class_num)
        print(out.shape)

        return out


if __name__ == '__main__':
    img = fluid.layers.data(name='img', shape=[3, 224, 224])

    net = VGG16(120)
    net.net(img)
