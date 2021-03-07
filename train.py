import tensorflow as tf
from glob import glob
import os
import scipy.misc
import numpy as np

batch_size = 15
stddev = 1.0
beta1 = 0.5
epochs = 1000
checkpoint_dir = 'D:\data\model'
learning_rate = 0.005


def linear(input,output_size,scope=None,stddev=0.02,bias_start=0.0):
    #print(input)
    shape= input.get_shape().as_list()#将tf转换为数组
    with tf.variable_scope(scope or "Linear"):
       #print(type(shape))
       #print(shape)
       matrix=tf.get_variable("Matrix",[shape[1],output_size],tf.float32,initializer=tf.random_normal_initializer(stddev=stddev))

       bias = tf.get_variable("bias",[output_size],initializer=tf.constant_initializer(bias_start))

        #print(bias)

       return tf.matmul(input,matrix)+bias


def generator(input):
   with tf.variable_scope('generator') as scope:
        s_h, s_w = 64,64
        s_h2, s_w2 = s_h / 2, s_w / 2
        s_h4, s_w4 = s_h / 4, s_w / 4
        s_h8, s_w8 = s_h / 8, s_w / 8
        s_h16, s_w16 = s_h / 16, s_w / 16

        z = linear(input,512*s_h16*s_w16,'generator_input_linear')
        z = tf.reshape(z, [-1, int (s_h16), int(s_w16), 512])
        z = tf.nn.relu(z)

        fifter2=tf.get_variable('fifter2',[5,5,256,512],initializer=tf.random_normal_initializer(stddev=stddev))
        conv2 = tf.nn.conv2d_transpose(z,fifter2,[batch_size,int(s_h8),int(s_w8),256],strides=[1,2,2,1])
        h1 = tf.contrib.layers.batch_norm(conv2,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h1 = tf.nn.relu(h1)

        fifter3=tf.get_variable('fifter3',[5,5,128,256],initializer=tf.random_normal_initializer(stddev=stddev))
        conv3 = tf.nn.conv2d_transpose(h1,fifter3,[batch_size,int(s_h4),int(s_w4),128],strides=[1,2,2,1])
        h2 = tf.contrib.layers.batch_norm(conv3,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h2 = tf.nn.relu(h2)

        fifter4=tf.get_variable('fifter4',[5,5,64,128],initializer=tf.random_normal_initializer(stddev=stddev))
        conv4 = tf.nn.conv2d_transpose(h2,fifter4,[batch_size,int(s_h2),int(s_w2),64],strides=[1,2,2,1])
        h3 = tf.contrib.layers.batch_norm(conv4,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h3 = tf.nn.relu(h3)

        fifter5=tf.get_variable('fifter5',[5,5,3,64],initializer=tf.random_normal_initializer(stddev=stddev))
        conv5 = tf.nn.conv2d_transpose(h3,fifter5,[batch_size,int(s_h),int(s_w),3], strides=[1,2,2,1])
        h4 = tf.contrib.layers.batch_norm(conv5,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)

        return tf.nn.sigmoid(h4)


def simple(z):#用来测试效果怎么样，和生成网络共享权重参数
   with tf.variable_scope('generator') as scope:
        scope.reuse_variables()
        s_h, s_w = 64,64
        s_h2, s_w2 = s_h / 2, s_w / 2
        s_h4, s_w4 = s_h / 4, s_w / 4
        s_h8, s_w8 = s_h / 8, s_w / 8
        s_h16, s_w16 = s_h / 16, s_w / 16

        z = linear(z,512*s_h16*s_w16,'generator_input_linear')
        z = tf.reshape(z, [-1, int (s_h16), int(s_w16), 512])
        z = tf.nn.relu(z)

        fifter2=tf.get_variable('fifter2',[5,5,256,512],initializer=tf.random_normal_initializer(stddev=stddev))
        conv2 = tf.nn.conv2d_transpose(z,fifter2,[batch_size,int(s_h8),int(s_w8),256],strides=[1,2,2,1])
        h1 = tf.contrib.layers.batch_norm(conv2,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h1 = tf.nn.relu(h1)

        fifter3=tf.get_variable('fifter3',[5,5,128,256],initializer=tf.random_normal_initializer(stddev=stddev))
        conv3 = tf.nn.conv2d_transpose(h1,fifter3,[batch_size,int(s_h4),int(s_w4),128],strides=[1,2,2,1])
        h2 = tf.contrib.layers.batch_norm(conv3,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h2 = tf.nn.relu(h2)

        fifter4=tf.get_variable('fifter4',[5,5,64,128],initializer=tf.random_normal_initializer(stddev=stddev))
        conv4 = tf.nn.conv2d_transpose(h2,fifter4,[batch_size,int(s_h2),int(s_w2),64],strides=[1,2,2,1])
        h3 = tf.contrib.layers.batch_norm(conv4,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h3 = tf.nn.relu(h3)

        fifter5=tf.get_variable('fifter5',[5,5,3,64],initializer=tf.random_normal_initializer(stddev=stddev))
        conv5 = tf.nn.conv2d_transpose(h3,fifter5,[batch_size,int(s_h),int(s_w),3], strides=[1,2,2,1])
        h4 = tf.contrib.layers.batch_norm(conv5,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)

        return tf.nn.sigmoid(h4)


def discriminator(image,reuse = False):

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        fifter1=tf.get_variable('fifter1',[5,5,3,64],initializer=tf.random_normal_initializer(stddev=stddev))
        h0=tf.nn.conv2d(image,fifter1,[1,2,2,1],padding="SAME")

        h0 = tf.contrib.layers.batch_norm(h0,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h0 = tf.maximum(h0,0.2*h0)

        fifter2=tf.get_variable('fifter2',[5,5,64,128],initializer=tf.random_normal_initializer(stddev=stddev))
        h1=tf.nn.conv2d(h0,fifter2,[1,2,2,1],padding="SAME")
        biases1=tf.get_variable('biases1',[128],initializer=tf.constant_initializer(0.0))
        h1=tf.reshape(tf.nn.bias_add(h1,biases1),h1.get_shape())
        h1 = tf.contrib.layers.batch_norm(h1,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h1 = tf.maximum(h1,0.2*h1)

        fifter3=tf.get_variable('fifter3',[5,5,128,256],initializer=tf.random_normal_initializer(stddev=stddev))
        h2=tf.nn.conv2d(h1,fifter3,[1,2,2,1],padding="SAME")
        biases2=tf.get_variable('biases2',[256],initializer=tf.constant_initializer(0.0))
        h2=tf.reshape(tf.nn.bias_add(h2,biases2),h2.get_shape())
        h2 = tf.contrib.layers.batch_norm(h2,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h2 = tf.maximum(h2,0.2*h2)

        fifter4=tf.get_variable('fifter4',[5,5,256,512],initializer=tf.random_normal_initializer(stddev=stddev))
        h3=tf.nn.conv2d(h2,fifter4,[1,2,2,1],padding="SAME")
        biases3=tf.get_variable('biases3',[512],initializer=tf.constant_initializer(0.0))
        h3=tf.reshape(tf.nn.bias_add(h3,biases3),h3.get_shape())
        h3 = tf.contrib.layers.batch_norm(h3,decay=0.9, updates_collections=None,epsilon=1e-5,scale=True,is_training=True)
        h3 = tf.maximum(h3,0.2*h3)

        h4 = linear(tf.reshape(h3,[-1,4*4*512]),1,'discriminator_output_linear')

        return h4,tf.nn.sigmoid(h4)



def get_deal_image(image_path, input_height=108, input_width=108,resize_height=64, resize_width=64):

    get_image = scipy.misc.imread(image_path).astype(np.float)
    return scipy.misc.imresize(get_image,[resize_height,resize_width])



def build_model(d_optim,g_optim,z,input):

    for epoch in range(0,epochs):
        pic_list= glob(os.path.join('D:\data\img_align_celeba',"*.jpg"))

        batch_idxs = len(pic_list)/batch_size
        counter = 1

        for idxs in range(0,3):

            counter = counter +1
            print(counter)

            batch_files= pic_list[idxs*batch_size:(idxs +1)*batch_size]
            batch=[get_deal_image(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z=np.random.uniform(-1,1,[batch_size,100]).astype(np.float32)

            with tf.Session() as sess:
                tf.global_variables_initializer().run() 

                _,d_loss_ = sess.run([d_optim,d_loss],feed_dict={z:batch_z, inputs:batch_images})

                _,g_loss_ = sess.run([g_optim,g_loss],feed_dict={z:batch_z})

                print('d_loss:%.8f,g_loss:%.8f' %(d_loss_,g_loss_))

                if np.mod(epoch,50)==1:

                    #每隔100张图片打印loss值

                    sample_z=np.random.uniform(-1,1,size=(batch_size,100))

                    samples = sess.run([simple(z)],feed_dict={z: sample_z}) 

                    print('100/d_loss:%.8f,100/g_loss:%.8f' %(d_loss_,g_loss_))

                    for idx,image in enumerate(samples[0]):
                        print(np.squeeze(image))
                        scipy.misc.imsave('D:\data\output\\train_{:02d}_{:04d}_1.png'.format(epoch, idx), np.squeeze(image))

    return sess


z = tf.placeholder(tf.float32, [None, 100], name='z')
inputs = tf.placeholder(tf.float32, [batch_size,64,64,3], name='real_images')

D_logits , D = discriminator(inputs,reuse = False)
D_logits_ , D_ = discriminator(generator(z), reuse = True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits,labels=tf.ones_like(D)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,labels=tf.zeros_like(D_)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,labels=tf.ones_like(D_)))

d_loss = d_loss_real + d_loss_fake 

t_vars = tf.trainable_variables()

d_vars = [var1 for var1 in t_vars if 'discriminator' in var1.name]
print(d_vars)

g_vars = [var2 for var2 in t_vars if 'generator' in var2.name]
print(g_vars)


d_optim=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(d_loss,var_list=d_vars)
g_optim=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(g_loss,var_list=g_vars)

sess = build_model(d_optim,g_optim,z,inputs)

saver=tf.train.Saver()
saver.save(sess,os.path.join(checkpoint_dir,'CGNN.model'))







        
        