# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import linecache
import numpy as np
import tensorflow as tf
import time
import math
"""
cut -d' ' -f 1 ./data/dbpedia.train | sort | uniq -c
40000 __label__1
40000 __label__10
40000 __label__11
40000 __label__12
40000 __label__13
40000 __label__14
40000 __label__2
40000 __label__3

"""
train_file = "data/dbpedia.train"
test_file = "data/dbpedia.test"
label_dict = {}
sku_dict = {}

max_window_size = 1000
batch_size = 500
emb_size = 128

# Parameters
learning_rate = 0.01
training_epochs = 1
display_step = 1

# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
# n_hidden_2 = 256 # 2nd layer number of features

def init_data(read_file):
    #0 is used for padding embedding
    label_cnt = 0
    sku_cnt = 1
    f = open(read_file,'r')
    for line in f:
        line = line.strip().split(' ')
        for i in line:
            if i.find('__label__') == 0:
                if i not in label_dict:
                    label_dict[i] = label_cnt
                    label_cnt += 1
            else:
                if i not in sku_dict:
                    sku_dict[i] = sku_cnt
                    sku_cnt += 1

def read_data(pos, batch_size, data_lst):
    batch = data_lst[pos:pos + batch_size]
    x = np.zeros((batch_size, max_window_size))
    mask = np.zeros((batch_size, max_window_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
        line = line.strip().split(' ')
        y.append(label_dict[line[0]])
        col_no = 0
        for i in line[1:]:
            if i in sku_dict:
                x[line_no][col_no] = sku_dict[i]
                mask[line_no][col_no] = 1
                col_no += 1
            if col_no >= max_window_size:
                break
        word_num[line_no] = col_no
        line_no += 1
    """
    x
        是一个(batch_size, max_window_size)，每一行代表一个user，每一行里面有若干个item_id，最多个数是max_window_size个item_id。

    word_num
        每个人购买的item_id个数不一定一直，每个人之前看了多少item_id，通过word_num来记录

    mask
        记录是标记，我x里面，哪些col_no的物品有实际的item_id的值

    y
        y就是这一次购买的item_id，长度是batch_size个记录

    整个一条数据的意义是，y，[x1,x2,x3]，我之前看了x1,x2,x3的3个物品，现在我这一次买了y这个物品

    这里最后返回的x里面，每个item的值是最后的item_id的编号，也就是从0开始的编号
    返回的y的值，也是从0开始的编号

    这里我用tensorflow我输入编号就像，转换成为one_hot vector，再进行一个所谓的矩阵相乘的操作，都被embeding_lookup来代替了

    """
    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, max_window_size, 1), word_num.reshape(batch_size, 1)

#========================
init_data(train_file)
n_classes = len(label_dict)
train_lst = linecache.getlines(train_file)
print("Class Num: ", n_classes)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1])),
    # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    #x = tf.nn.dropout(x, 0.8)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #dlayer_1 = tf.nn.dropout(layer_1, 0.5)
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # return out_layer
    return layer_1

embedding = {
    'input':tf.Variable(tf.random_uniform([len(sku_dict)+1, emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}

emb_mask = tf.placeholder(tf.float32, shape=[None, max_window_size, 1])
# 这个用户看了有之前购买过多少item_id
word_num = tf.placeholder(tf.float32, shape=[None, 1])

x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])
y_batch = tf.placeholder(tf.int64, [None, 1])

"""

返回的tensor的shape = shape(ids) + shape(params)[1:]
e.g.: ids.shape()=[A,B,C];
      params.shape()=[X,Y];
那么:
      results.shape()=[A,B,C,Y]

这里embedding的shape=[item_id_num,emb_size],这个是param
x_batch.shape=[batch_size,max_window_size]，这个是ids

所以结果的shape就是[batch_size,max_window_size,emb_size]
也就是在每个ids上面，又增加了一个维度，这个维度上面的值是embedding

想想一个平面，我这个平面是一个2维的，也就是我的ids的维度，然后每个坐标
上面有一只值，这个值映射到embedding上面的一个vector，然后我这个look up
操作，就是让我的这个平面上的点，分别长出了对应embedding的vector

"""

# input_embedding # None*max_window_size*emb_size
input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
"""
element wise multy,embedding_mask的维度是[None, max_window_size, 1]
None*max_window_size*emb_size*[None, max_window_size, 1]
也就是因为自己输入的x_batch里面有一些元素是0，这些本来应该代表没有任何值的，但是
因为embedding look up对于ids为0的那些也会去embedding里面去找，所以这里embedding的size就是[len(sku_dict)+1, emb_size]，也就是故意多出来1个维度

这里多出来的这个维度，会和embedding_mask相乘，就会变成0，所以这些也就不会有影响了
然后reduce_sum把axis=1的这维度，也就是max_window_size这个维度上面进行相加
最后再除以word_num，就获得了一个embedding的avg

reduce sum之后变成，None*emb_size
现在的意义，就是我每个user，我所有看过的item_id进行embedding，然后我把这些embedding进行avg，就获得了用这些item_id的embedding来描述这个user了
"""
project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding,emb_mask), 1),word_num)

# Construct model
# weights=embed_size*hidden_layer_unit
pred = multilayer_perceptron(project_embedding, weights, biases)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([n_classes, n_hidden_1],
                        stddev=1.0 / math.sqrt(n_hidden_1)))
nce_biases = tf.Variable(tf.zeros([n_classes]))
"""
这里nce_loss还是不太懂怎么用，这里的n_classes，对应的label num，也就是对应的我所有购买的item_id的数量，

这里的nce_weight的维度是[n_classes, n_hidden_1],其实也就是我预测n个物品的embedding

这里的nce_loss，应该做的事情就是，我input的是我历史上购买的[x1,x2,x3]的物品的embedding的值，导致了我这次购买了y这个物品，这里的y也是一个编号

我这里应该就是用nce进行一个负采样，然后训练最后的loss

"""

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=y_batch,
                     inputs=pred,
                     num_sampled=10,
                     num_classes=n_classes))

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases

init = tf.global_variables_initializer()
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    start_time = time.time()
    total_batch = int(len(train_lst) / batch_size)
    print("total_batch of training data: ", total_batch)
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
            x, y, batch_mask, word_number = read_data(i * batch_size, batch_size, train_lst)
            _,c = sess.run([optimizer, cost], feed_dict={x_batch: x, emb_mask: batch_mask, word_num: word_number, y_batch: y})
            #print("Epoch %d Batch %d Elapsed time %fs" %(epoch, i, time.time() - start_time))
            # Compute average loss
            avg_cost += c / total_batch
            # correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [batch_size]))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print("Accuracy:", accuracy.eval({x_batch: x, y_batch: y, emb_mask: batch_mask, word_num: word_number}))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))

    # Test model
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [batch_size]))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    test_lst = linecache.getlines(test_file)
    total_batch = int(len(test_lst) / batch_size)
    final_accuracy = 0
    for i in range(total_batch):
        x, y, batch_mask, word_number = read_data(i*batch_size, batch_size, test_lst)
        batch_accuracy = accuracy.eval({x_batch: x, y_batch: y, emb_mask: batch_mask, word_num: word_number})
        print("Batch Accuracy: ", batch_accuracy)
        final_accuracy += batch_accuracy
    print("Final Accuracy: ", final_accuracy * 1.0 / total_batch)
