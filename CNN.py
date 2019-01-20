from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 从压缩包采集数据


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 定义Weight变量，输入shape，返回变量的一些参数。 ???
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    # 使用tf.truncted_normal产生随机变量来进行初始化
    return tf.Variable(initial)


# 定义bias变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)     # 常量函数初始化
    return tf.Variable(initial)


"""
定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，
strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。
"""


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 使用最大值池化
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 占位符
# define placeholder for inputs to network 定义输入的placeholder
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)      # dropout的placeholder
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# 对输入的图片reshape:-1:不考虑输入的图片的维度 像素28X28 channel1 灰白图像
# print(x_image.shape)  # [n_samples, 28,28,1]


# conv1 layer
"""""
Weight:卷积核patch大小：5X5 channel:1 输出32个featuremap
定义bias，它的大小是32个长度，因此我们传入它的shape为[32]
定义好了Weight和bias，我们就可以定义卷积神经网络的第一个卷积层h_conv1=conv2d(x_image,W_conv1)+b_conv1,同时我们对h_conv1进行非线性处理，
也就是激活函数来处理，这里我们用的是tf.nn.relu（修正线性单元）来处理，要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，
只是厚度变厚了，因此现在的输出大小就变成了28x28x32
"""""
#with tf.name_scope('layer1'):
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32 非线性处理（激活函数）
h_pool1 = max_pool_2x2(h_conv1)                           # 池化缩小长宽output size 14x14x32


# conv2 layer
#with tf.name_scope('layer2'):
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64


# fc1 layer
# 进入全连接层前通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
#with tf.name_scope('fc1'):
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    # 考虑过拟合的问题，加一个dropout处理


# fc2 layer 输入是1024，最后输出10(因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值
#with tf.name_scope('fc2'):
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)    # softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类


# the error between prediction and real data 利用交叉熵损失函数来定义我们的cost function
#with tf.name_scope('loss'):
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
# tf.summary.scalar('loss', cross_entropy)
# 用tf.train.AdamOptimizer()作为我们的优化器进行优化，更新参数（权值）使我们的cross_entropy最小
#with tf.name_scope('train'):
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
#merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter("logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver() # 定义save

# 训练1000次，每50次输出正确率
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})    # feed_dict喂数据
    if i % 50 == 0:
        accuracy = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])
        tf.summary.scalar('accuracy', accuracy)
        # print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        print(accuracy)
        # result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        # writer.add_summary(result, i)
    saver.save(sess, 'model/')  # 模型储存位置 当前目录model文件夹中
