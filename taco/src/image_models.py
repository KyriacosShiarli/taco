import tensorflow as tf
from tensorflow import space_to_batch, batch_to_space
import numpy as np

def conv_layers(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(5, (5,5),strides=(1,1), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(3, (3,3),strides=(1,1), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output



def conv_layers_10_5_3_sub(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (5,5),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(5, (5,5),strides=(2,2), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(3, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output

def conv_layers_10_15_20_sub(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (5,5),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(15, (5,5),strides=(2,2), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(20, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output

def conv_layers_10_5_5_sub(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (5,5),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(5, (5,5),strides=(2,2), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(5, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output

def conv_layers_10_5_3(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (5,5),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(5, (5,5),strides=(2,2), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output



def conv_layers_minimal(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (5,5),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(1, (1,1),strides=(1,1), activation='relu', padding= 'same', name='conv_2')(conv_1)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_2)
    #output = tf.cast(output, tf.float64)
    return output


def conv_layers_10_5_1_sub(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (3,3),strides=(2,2), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(5, (3,3),strides=(2,2), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output

def conv_layers_10_5_1(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(10, (3,3),strides=(1,1), activation= 'relu', padding= 'same', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), activation='relu', padding= 'same', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output

def conv_layers_dilated_v2(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2

    net = tf.keras.layers.Conv2D(6, (3,3),strides=(1,1), dilation_rate=int(rate ** 0),activation= 'relu', padding="same", name="dil_conv_1")(x)
    print(net.get_shape())
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 1),activation= 'relu', padding="same", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(3 , (3,3),strides=(1,1), dilation_rate=int(rate ** 2),activation= 'relu', padding="same", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(2, (3,3),strides=(1,1), dilation_rate=int(rate ** 3),activation= 'relu', padding="same", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(1, (3,3),strides=(1,1), dilation_rate=int(rate ** 4),activation= 'relu', padding="same", name="dil_conv_1")(net)
    print(net.get_shape())
    output = tf.layers.flatten(net)
    print(output.get_shape())
    return output

def conv_layers_dilated_v2_3(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2

    net = tf.keras.layers.Conv2D(10, (3,3),strides=(1,1), dilation_rate=int(rate ** 0),activation= 'relu', padding="same", name="dil_conv_1")(x)
    print(net.get_shape())
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 1),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 2),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 3),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 4),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), dilation_rate=int(rate ** 4), activation='relu',padding="valid", name="dil_conv_1")(net)

    print(net.get_shape())
    output = tf.layers.flatten(net)
    print(output.get_shape())
    return output

def conv_layers_dilated_v2_4(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2

    net = tf.keras.layers.Conv2D(10, (3,3),strides=(1,1), dilation_rate=int(rate ** 0),activation= 'relu', padding="same", name="dil_conv_1")(x)
    print(net.get_shape())
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 1),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 2),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), dilation_rate=int(rate ** 3), activation='relu',padding="valid", name="dil_conv_1")(net)

    print(net.get_shape())
    output = tf.layers.flatten(net)
    print(output.get_shape())
    return output

def conv_layers_dilated_v2_2(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2

    net = tf.keras.layers.Conv2D(20, (3,3),strides=(1,1), dilation_rate=int(rate ** 0),activation= 'relu', padding="same", name="dil_conv_1")(x)
    print(net.get_shape())
    net = tf.keras.layers.Conv2D(15, (3,3),strides=(1,1), dilation_rate=int(rate ** 1),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(10, (3,3),strides=(1,1), dilation_rate=int(rate ** 2),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(5, (3,3),strides=(1,1), dilation_rate=int(rate ** 3),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    net = tf.keras.layers.Conv2D(1, (3,3),strides=(1,1), dilation_rate=int(rate ** 4),activation= 'relu', padding="valid", name="dil_conv_1")(net)
    print(net.get_shape())
    output = tf.layers.flatten(net)
    print(output.get_shape())
    return output



def conv_layers_dilated_1(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,5]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters4 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters5 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters6 = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="SAME", name="dil_conv_1")
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="SAME", name="dil_conv_2")
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="SAME", name="dil_conv_3")
    net = tf.nn.atrous_conv2d(net, filters4, rate ** 3, padding="SAME", name="dil_conv_4")
    net = tf.nn.atrous_conv2d(net, filters5, rate ** 4, padding="SAME", name="dil_conv_5")
    net = tf.nn.atrous_conv2d(net, filters6, rate ** 5, padding="SAME", name="dil_conv_6")
    output = tf.layers.flatten(net)
    return output

def conv_layers_dilated_2(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,5]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,5,3]), dtype=tf.float32)
    filters4 = tf.Variable(tf.random_normal([3,3,3,3]), dtype=tf.float32)
    filters5 = tf.Variable(tf.random_normal([3,3,3,1]), dtype=tf.float32)
    filters6 = tf.Variable(tf.random_normal([3,3,1,1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="SAME", name="dil_conv_1")
    print(net.get_shape())
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="SAME", name="dil_conv_2")
    print(net.get_shape())
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="SAME", name="dil_conv_3")
    print(net.get_shape())
    net = tf.nn.atrous_conv2d(net, filters4, rate ** 3, padding="SAME", name="dil_conv_4")
    print(net.get_shape())
    net = tf.nn.atrous_conv2d(net, filters5, rate ** 4, padding="SAME", name="dil_conv_5")
    print(net.get_shape())
    net = tf.nn.atrous_conv2d(net, filters6, rate ** 5, padding="SAME", name="dil_conv_6")
    print(net.get_shape())
    output = tf.layers.flatten(net)
    return output

def conv_layers_dilated_3(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,20]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,20,20]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,20,10]), dtype=tf.float32)
    filters4 = tf.Variable(tf.random_normal([3,3,10,10]), dtype=tf.float32)
    filters5 = tf.Variable(tf.random_normal([3,3,10,1]), dtype=tf.float32)
    filters6 = tf.Variable(tf.random_normal([3,3,1,1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="SAME", name="dil_conv_1")
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="SAME", name="dil_conv_2")
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="SAME", name="dil_conv_3")
    net = tf.nn.atrous_conv2d(net, filters4, rate ** 3, padding="SAME", name="dil_conv_4")
    net = tf.nn.atrous_conv2d(net, filters5, rate ** 4, padding="SAME", name="dil_conv_5")
    net = tf.nn.atrous_conv2d(net, filters6, rate ** 5, padding="SAME", name="dil_conv_6")
    output = tf.layers.flatten(net)
    return output

def conv_layers_dilated_5(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,10]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,10,5]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,5,1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="SAME", name="dil_conv_1")
    net = tf.nn.relu(net,name="relu_1")
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="SAME", name="dil_conv_2")
    net = tf.nn.relu(net,name="relu_2")
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="SAME", name="dil_conv_3")
    net = tf.nn.relu(net,name="relu_3")

    output = tf.layers.flatten(net)
    return output

def conv_layers_dilated_6(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,10]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,10,6]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,6,3]), dtype=tf.float32)
    filters4 = tf.Variable(tf.random_normal([3, 3, 3, 1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="VALID", name="dil_conv_1")
    net = tf.nn.relu(net,name="relu_1")
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="VALID", name="dil_conv_2")
    net = tf.nn.relu(net,name="relu_2")
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="VALID", name="dil_conv_3")
    net = tf.nn.relu(net,name="relu_3")
    net = tf.nn.atrous_conv2d(net, filters4, rate ** 3, padding="VALID", name="dil_conv_4")
    net = tf.nn.relu(net,name="relu_4")
    output = tf.layers.flatten(net)
    return output

def conv_layers_dilated_4(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 2
    filters1 = tf.Variable(tf.random_normal([3,3,3,5]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,5,3]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,3,1]), dtype=tf.float32)

    net = tf.nn.atrous_conv2d(x, filters1, rate ** 0, padding="SAME", name="dil_conv_1")
    net = tf.nn.atrous_conv2d(net, filters2, rate ** 1, padding="SAME", name="dil_conv_2")
    net = tf.nn.atrous_conv2d(net, filters3, rate ** 2, padding="SAME", name="dil_conv_3")

    output = tf.layers.flatten(net)
    return output

def conv_layers_2(x, bs, max_seq_len, im_dim):
    #x = tf.cast(x, tf.float32)
    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    rate = 3
    rem_wid, rem_hei = rate * 2 - (wid % (rate * 2)) , rate * 2 - (hei % (rate * 2))
    pad = tf.constant([[np.floor(rem_hei / 2), np.ceil(rem_hei / 2)], [np.floor(rem_wid / 2), np.ceil(rem_wid / 2)]])
    pad = tf.cast(pad, tf.int32)
    filters1 = tf.Variable(tf.random_normal([3,3,3,5]), dtype=tf.float32)
    filters2 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    filters3 = tf.Variable(tf.random_normal([3,3,5,5]), dtype=tf.float32)
    net = space_to_batch(x,paddings=pad,block_size=rate)
    net = tf.nn.conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME", name="dil_conv_1")
    # print "dil_conv_1"
    # print net.get_shape()
    # net = tf.nn.conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME", name="dil_conv_2")
    # print "dil_conv_2"
    # print net.get_shape()
    net = tf.nn.conv2d(net, filters3, strides=[1, 1, 1, 1], padding="SAME", name="dil_conv_3")
    # print "dil_conv_3"
    # print net.get_shape()
    net = batch_to_space(net, crops=pad, block_size=rate)
    # print "final_output"
    # print net.get_shape()
    output = tf.layers.flatten(net)
    # print "output"
    # print output.get_shape()
    return output


def conv_layers_20_10_1(x, bs, max_seq_len, im_dim):

    #x = tf.cast(x, tf.float32)

    wid, hei, chan = im_dim[0], im_dim[1], im_dim[2]
    try:
        x = tf.reshape(x, [bs * max_seq_len, wid, hei, chan])
    except:
        print('image dimensions not compatible')
        raise
    conv_1 = tf.keras.layers.Conv2D(20, (3,3),strides=(1,1), activation= 'relu', padding= 'valid', name='conv_1')(x)
    conv_2 = tf.keras.layers.Conv2D(10, (3,3),strides=(1,1), activation='relu', padding= 'valid', name='conv_2')(conv_1)
    conv_3 = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv_3')(
        conv_2)
    #output = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2), name='max_pool')(conv_2)
    output = tf.layers.flatten(conv_3)
    #output = tf.cast(output, tf.float64)
    return output


image_models = {}
image_models['conv_layers'] = conv_layers
image_models['conv_layers_dilated_1'] = conv_layers_dilated_1
image_models['conv_layers_dilated_2'] = conv_layers_dilated_2
image_models['conv_layers_dilated_3'] = conv_layers_dilated_3
image_models['conv_layers_dilated_4'] = conv_layers_dilated_4
image_models['conv_layers_dilated_5'] = conv_layers_dilated_5
image_models['conv_layers_dilated_6'] = conv_layers_dilated_6
image_models['conv_layers_2'] = conv_layers_2
image_models['conv_layers_10_5_1'] = conv_layers_10_5_1
image_models['conv_layers_10_15_20_sub'] = conv_layers_10_15_20_sub
image_models['conv_layers_10_5_5_sub'] = conv_layers_10_5_5_sub
image_models['conv_layers_10_5_3'] = conv_layers_10_5_3

image_models['conv_layers_10_5_3_sub'] = conv_layers_10_5_3_sub
image_models['conv_layers_minimal'] = conv_layers_minimal
image_models['conv_layers_10_5_1_sub'] = conv_layers_10_5_1_sub
image_models['conv_layers_20_10_1'] = conv_layers_20_10_1
image_models['conv_layers_dilated_v2'] = conv_layers_dilated_v2
image_models['conv_layers_dilated_v2_2'] = conv_layers_dilated_v2_2
image_models['conv_layers_dilated_v2_3'] = conv_layers_dilated_v2_3
image_models['conv_layers_dilated_v2_4'] = conv_layers_dilated_v2_4
