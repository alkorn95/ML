import tensorflow as tf
import os
import random

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, 1)
  image_resized = tf.image.resize_images(image_decoded, [32, 32])
  return image_resized, label

path="D:/2/Train/"
tr_filenames = []
tr_labels = []
te_filenames = []
te_labels = []
for i in range(10):
  for j in range(120):
    if j%6!=0:
      tr_filenames.append(path+str(i)+"/"+str(j)+".jpg")
      tr_labels.append(i)
    else:
      te_filenames.append(path+str(i)+"/"+str(j)+".jpg")
      te_labels.append(i)


tr_dataset = tf.data.Dataset.from_tensor_slices((tr_filenames, tr_labels))#обучающая выборка
tr_dataset = tr_dataset.map(_parse_function)
te_dataset = tf.data.Dataset.from_tensor_slices((te_filenames, te_labels))#тестовая выборка
te_dataset = te_dataset.map(_parse_function)

learning_rate = 0.001
num_steps = 1000
batch_size = 20
display_step = 1

n_input = 1024 
n_classes = 10 
dropout = 0.25 

sess = tf.Session()
tr_dataset=tr_dataset.shuffle(buffer_size=10000)
tr_dataset = tr_dataset.batch(batch_size)
tr_dataset=tr_dataset.repeat(num_steps)
iterator = tr_dataset.make_one_shot_iterator()
X, Y = iterator.get_next()

te_dataset = te_dataset.batch(1)
iterator1 = te_dataset.make_one_shot_iterator()
X1, Y1 = iterator1.get_next()

def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 32, 32, 1])
        conv1 = tf.layers.conv2d(x, 15, 5, activation=tf.nn.leaky_relu)                                 
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 20, 5, activation=tf.nn.leaky_relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 50)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
        out = tf.nn.leaky_relu(out)        
        out = tf.nn.softmax(out) if not is_training else out

    return out

logits_train = conv_net(X, n_classes, dropout, reuse=False, is_training=True)
logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_train,  labels=tf.one_hot(indices=tf.cast(Y, tf.int32),depth=10)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(tf.one_hot(indices=tf.cast(Y, tf.int32),depth=10), 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


logits_test1 = conv_net(X1, n_classes, dropout, reuse=True, is_training=False)
correct_pred1 = tf.equal(tf.argmax(logits_test1, 1), tf.argmax(tf.one_hot(indices=tf.cast(Y1, tf.int32),depth=10), 1))
accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
init = tf.global_variables_initializer()

sess.run(init)

for step in range(1, num_steps + 1):
  sess.run(train_op)
    

  if step % display_step == 0 or step == 1:
    loss, acc = sess.run([loss_op, accuracy])
    print("Step " + str(step) + ", Batch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")
t=0
f=0
for step in range(0, 200):
 
  if step % display_step == 0 or step == 1:
    acc = sess.run(accuracy1)
    if acc == 1:
      t=t+1
    else:
      f=f+1
    print("Step " + str(step) +  ", Accuracy= " + \
              "{:.3f}".format(acc))
    print(str(t)+" "+str(f)+" "+str(t/(t+f)))


