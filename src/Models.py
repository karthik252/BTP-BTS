import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D, BatchNormalization, Lambda, Multiply, Conv1D, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from codeUtils import *
K.set_image_data_format('channels_last')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_variable(name, shape): 
    return tf.compat.v1.get_variable(name=name, shape=shape, initializer=GlorotUniform(), regularizer=None)

class VFR_SM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VFR_SM, self).__init__(**kwargs)

    def build(self, input_shape):
        a, s, c = input_shape[1:4]
        filters = input_shape[4]
        self.kernel1_a = self.add_weight(shape=[filters, 1, 1, a, a//4], trainable=True, initializer=GlorotUniform(), name='kernel1_a')
        self.kernel1_s = self.add_weight(shape=[filters, 1, 1, s, s//4], trainable=True, initializer=GlorotUniform(), name='kernel1_s')
        self.kernel1_c = self.add_weight(shape=[filters, 1, 1, c, c//4], trainable=True, initializer=GlorotUniform(), name='kernel1_c')
        self.kernel2_a = self.add_weight(shape=[filters, 1, 1, a//4, a], trainable=True, initializer=GlorotUniform(), name='kernel2_a')
        self.kernel2_s = self.add_weight(shape=[filters, 1, 1, s//4, s], trainable=True, initializer=GlorotUniform(), name='kernel2_s')
        self.kernel2_c = self.add_weight(shape=[filters, 1, 1, c//4, c], trainable=True, initializer=GlorotUniform(), name='kernel2_c')
        super(VFR_SM, self).build(input_shape)

    def call(self, inputs):
        x_scan1 = tf.transpose(inputs,[4,0,1,2,3])
        mat = tf.scan(self.__f, (x_scan1, self.kernel1_a, self.kernel1_s, self.kernel1_c, self.kernel2_a, self.kernel2_s, self.kernel2_c,), initializer=tf.fill(tf.shape(inputs)[:-1], 0.0))     
        return tf.transpose(mat,[1,2,3,4,0])

    def __f(self, _, inp):
        h1 = inp[0]
        kernel1_a = inp[1] 
        kernel1_s = inp[2] 
        kernel1_c = inp[3] 
        kernel2_a = inp[4] 
        kernel2_s = inp[5] 
        kernel2_c = inp[6] 

        conv_axis1 = tf.reduce_mean(h1,axis = [1,2], keepdims = True)
        conv_axis1 = tf.nn.relu(tf.nn.conv2d(conv_axis1,kernel1_c,[1,1,1,1],padding = "SAME"))
        conv_axis1 = tf.nn.sigmoid(tf.nn.conv2d(conv_axis1,kernel2_c,[1,1,1,1],padding = "SAME"))

        conv_axis2 = tf.reduce_mean(h1,axis = [1,3], keepdims = True)
        conv_axis2 = tf.transpose(conv_axis2,[0,1,3,2])
        conv_axis2 = tf.nn.relu(tf.nn.conv2d(conv_axis2,kernel1_s,[1,1,1,1],padding = "SAME"))
        conv_axis2 = tf.nn.sigmoid(tf.nn.conv2d(conv_axis2,kernel2_s,[1,1,1,1],padding = "SAME"))
        conv_axis2 = tf.transpose(conv_axis2,[0,1,3,2])

        conv_axis3 = tf.reduce_mean(h1,axis = [2,3], keepdims = True)
        conv_axis3 = tf.transpose(conv_axis3,[0,2,3,1])
        conv_axis3 = tf.nn.relu(tf.nn.conv2d(conv_axis3,kernel1_a,[1,1,1,1],padding = "SAME"))
        conv_axis3 = tf.nn.sigmoid(tf.nn.conv2d(conv_axis3,kernel2_a,[1,1,1,1],padding = "SAME"))
        conv_axis3 = tf.transpose(conv_axis3,[0,3,1,2])

        mat = tf.multiply(conv_axis1,h1)
        mat = tf.multiply(conv_axis2,mat)
        mat = tf.multiply(conv_axis3,mat)

        return mat
    
def SM3D():# add reguralizer
    inputs = Input((img_rows_trim, img_cols_trim, img_depth_trim, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)

    pool1 = MaxPooling3D(pool_size=(3, 3, 3))(conv3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    pool2 = MaxPooling3D(pool_size=(3, 3, 3))(conv5)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    deconv1 = Conv3DTranspose(64, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(conv7)
    cc1 = concatenate([deconv1, conv5], axis=-1)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(cc1)
    conv9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)

    deconv2 = Conv3DTranspose(32, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(conv9)
    cc2 = concatenate([deconv2, conv3], axis=-1)
    conv10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(cc2)
    conv11 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv10)

    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv11)
    OUT = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='OUT')(conv12)

    ds1deconv = Conv3DTranspose(64, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(pool1)
    ds1conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ds1deconv)
    DS1 = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='DS1')(ds1conv)

    ds2deconv = Conv3DTranspose(128, (3, 3, 3), strides=(9, 9, 11), output_padding=(6, 6, 0))(pool2) # changed 128 to 64
    ds2conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ds2deconv)
    DS2 = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='DS2')(ds2conv)
    model = Model(inputs=[inputs], outputs=[DS1, DS2, OUT])
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000199), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', dice_coef])
    return model

def SWSM3D():
    inputs = Input((img_rows_trim, img_cols_trim, img_depth_trim, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)

    conv3 = VFR_SM(name='VFR1')(conv3)
    pool1 = MaxPooling3D(pool_size=(3, 3, 3))(conv3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    conv5 = VFR_SM(name='VFR2')(conv5)
    pool2 = MaxPooling3D(pool_size=(3, 3, 3))(conv5)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    conv7 = VFR_SM(name='VFR3')(conv7)
    deconv1 = Conv3DTranspose(64, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(conv7)
    cc1 = concatenate([deconv1, conv5], axis=-1)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(cc1)
    conv9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)

    conv9 = VFR_SM(name='VFR4')(conv9)
    deconv2 = Conv3DTranspose(32, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(conv9)
    cc2 = concatenate([deconv2, conv3], axis=-1)
    conv10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(cc2)
    conv11 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv10)

    conv11 = VFR_SM( name='VFR5')(conv11)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv11)
    OUT = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='OUT')(conv12)

    ds1deconv = Conv3DTranspose(64, (3, 3, 3), strides=(3, 3, 3), output_padding=(0, 0, 2))(pool1)
    ds1conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ds1deconv)
    DS1 = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='DS1')(ds1conv)

    ds2deconv = Conv3DTranspose(128, (3, 3, 3), strides=(9, 9, 11), output_padding=(6, 6, 0))(pool2)
    ds2conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(ds2deconv)
    DS2 = Conv3D(5, (1, 1, 1), activation='softmax', padding='same', name='DS2')(ds2conv)

    model = Model(inputs=[inputs], outputs=[DS1, DS2, OUT])
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000199), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy', dice_coef])
    return model

def reluConv(filters, layer, kernel = (3, 3, 3)):
    temp = Conv3D(filters, kernel,activation = 'relu', padding = 'same')(layer)
    return BatchNormalization()(temp)

def VFR(layers, filters):
    # The Conv output is a 5D tensor in the shape (batch_size, conv_dim1, conv_dim2, conv_dim3, channels)

    layer_m1 = layers[0]
    layer_m2 = layers[1]
    layer_m3 = layers[2]
    
    shape = list(layer_m1.shape)
    shapeTensor = tf.shape(layer_m1)

    h1 = concatenate([layer_m1, layer_m2, layer_m3], axis = 1)
    h1 = Lambda(lambda x: tf.reduce_mean(x, axis = [2, 3],keepdims= False))(h1) # 1D vector
    
    a1_axis3 = Conv1D(filters, shape[1] + 1, 2, activation = 'sigmoid')(h1)
    a1_axis3 = Conv1D(filters, int(shape[1]*3/4), 1, activation = 'relu', padding = 'same')(a1_axis3)
    a2_axis3 = Conv1D(filters, shape[1] + 1, 2, activation = 'sigmoid')(h1)
    a2_axis3 = Conv1D(filters, int(shape[1]*3/4), 1, activation = 'relu', padding = 'same')(a2_axis3)
    a3_axis3 = Conv1D(filters, shape[1] + 1, 2, activation = 'sigmoid')(h1)
    a3_axis3 = Conv1D(filters, int(shape[1]*3/4), 1, activation = 'relu', padding = 'same')(a3_axis3)

    h2 = concatenate([layer_m1, layer_m2, layer_m3], axis = 2)
    h2 = Lambda(lambda x: tf.reduce_mean(x, axis = [1, 3],keepdims= False))(h2) # 1D vector

    a1_axis2 = Conv1D(filters, shape[2] + 1, 2, activation = 'sigmoid')(h2)
    a1_axis2 = Conv1D(filters, int(shape[2]*3/4), 1, activation = 'relu', padding = 'same')(a1_axis2)
    a2_axis2 = Conv1D(filters, shape[2] + 1, 2, activation = 'sigmoid')(h2)
    a2_axis2 = Conv1D(filters, int(shape[2]*3/4), 1, activation = 'relu', padding = 'same')(a2_axis2)
    a3_axis2 = Conv1D(filters, shape[2] + 1, 2, activation = 'sigmoid')(h2)
    a3_axis2 = Conv1D(filters, int(shape[2]*3/4), 1, activation = 'relu', padding = 'same')(a3_axis2)

    h3 = concatenate([layer_m1, layer_m2, layer_m3], axis = 3)
    h3 = Lambda(lambda x: tf.reduce_mean(x, axis = [1, 2],keepdims= False))(h3) # 1D vector

    a1_axis1 = Conv1D(filters, shape[3] + 1, 2, activation = 'sigmoid')(h3)
    a1_axis1 = Conv1D(filters, int(shape[3]*3/4), 1, activation = 'relu', padding = 'same')(a1_axis1)
    a2_axis1 = Conv1D(filters, shape[3] + 1, 2, activation = 'sigmoid')(h3)
    a2_axis1 = Conv1D(filters, int(shape[3]*3/4), 1, activation = 'relu', padding = 'same')(a2_axis1)
    a3_axis1 = Conv1D(filters, shape[3] + 1, 2, activation = 'sigmoid')(h3)
    a3_axis1 = Conv1D(filters, int(shape[3]*3/4), 1, activation = 'relu', padding = 'same')(a3_axis1)

    a1_axis3 = Reshape([shape[1], 1, 1, filters])(a1_axis3)
    a1_axis3 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a1_axis3)
    a2_axis3 = Reshape([shape[1], 1, 1, filters])(a2_axis3)
    a2_axis3 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a2_axis3)
    a3_axis3 = Reshape([shape[1], 1, 1, filters])(a3_axis3)
    a3_axis3 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a3_axis3)

    a1_axis2 = Reshape([1, shape[2], 1, filters])(a1_axis2)
    a2_axis2 = Reshape([1, shape[2], 1, filters])(a2_axis2)
    a2_axis2 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a2_axis2)
    a3_axis2 = Reshape([1, shape[2], 1, filters])(a3_axis2)
    a3_axis2 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a3_axis2)
    
    a1_axis1 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a1_axis1)
    a2_axis1 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a2_axis1)
    a3_axis1 = Lambda(lambda x: tf.broadcast_to(x, shapeTensor))(a3_axis1)

    op1 = multiply([layer_m1, a1_axis1])
    op1 = multiply([op1, a1_axis2])
    op1 = multiply([op1, a1_axis3])

    op2 = multiply([layer_m2, a2_axis1])
    op2 = multiply([op2, a2_axis2])
    op2 = multiply([op2, a2_axis3])

    op3 = multiply([layer_m3, a3_axis1])
    op3 = multiply([op3, a3_axis2])
    op3 = multiply([op3, a3_axis3])

    return [op1, op2, op3]

def maxPool(layer):
    return MaxPooling3D(pool_size=(2, 2, 2))(layer)

def deConv(filters, layer, kernel = (2, 2, 2)):
    return Conv3DTranspose(filters, kernel, strides = (2,2,2))(layer)

def softConv(filters, layer, kernel = (1, 1, 1), name='softConv'):
    return Conv3D(filters, kernel, activation = 'softmax', padding = 'same', name=name)(layer)

def SWMM3D():
    
    inputs_m1 = Input((img_rows_trim, img_cols_trim, img_depth_trim, 1))
    inputs_m2 = Input((img_rows_trim, img_cols_trim, img_depth_trim, 1))
    inputs_m3 = Input((img_rows_trim, img_cols_trim, img_depth_trim, 1))
    print(tf.shape(inputs_m1))

    conv1_m1 = reluConv(32, inputs_m1)
    conv1_m2 = reluConv(32, inputs_m2)
    conv1_m3 = reluConv(32, inputs_m3)
    print(tf.shape(conv1_m1))

    conv1_m1 = reluConv(64, conv1_m1)
    conv1_m2 = reluConv(64, conv1_m2)
    conv1_m3 = reluConv(64, conv1_m3)

    conv1_m1 = reluConv(64, conv1_m1)
    conv1_m2 = reluConv(64, conv1_m2)
    conv1_m3 = reluConv(64, conv1_m3)

    vfr1_m1, vfr1_m2, vfr1_m3 = VFR([conv1_m1, conv1_m2, conv1_m3], 64)
    pool1_m1 = maxPool(vfr1_m1)
    pool1_m2 = maxPool(vfr1_m2)
    pool1_m3 = maxPool(vfr1_m3)

    #-------------------------------------------- DS1
    deconv_ds1 = concatenate([deConv(64, pool1_m1), deConv(64, pool1_m2), deConv(64, pool1_m3)], axis = 4)
    conv_ds1 = reluConv(32, deconv_ds1)
    out_ds1 = softConv(1, conv_ds1, name='DS1')
    #------------------------------------------------
    
    conv2_m1 = reluConv(128, pool1_m1)
    conv2_m2 = reluConv(128, pool1_m2)
    conv2_m3 = reluConv(128, pool1_m3)

    conv2_m1 = reluConv(128, conv2_m1)
    conv2_m2 = reluConv(128, conv2_m2)
    conv2_m3 = reluConv(128, conv2_m3)

    vfr2_m1, vfr2_m2, vfr2_m3 = VFR([conv2_m1, conv2_m2, conv2_m3], 128)
    pool2_m1 = maxPool(vfr2_m1)
    pool2_m2 = maxPool(vfr2_m2)
    pool2_m3 = maxPool(vfr2_m3) 
    
    #-------------------------------------------- DS2
    deconv_ds2_m1 = deConv(128, pool2_m1)
    deconv_ds2_m2 = deConv(128, pool2_m2)
    deconv_ds2_m3 = deConv(128, pool2_m3)

    deconv_ds2 = concatenate([deConv(128, deconv_ds2_m1), deConv(128, deconv_ds2_m2), deConv(128, deconv_ds2_m3)], axis = 4)
    conv_ds2 = reluConv(32, deconv_ds2)
    out_ds2 = softConv(1, conv_ds2, name='DS2')
    #------------------------------------------------

    conv3_m1 = reluConv(256, pool2_m1)
    conv3_m2 = reluConv(256, pool2_m2)
    conv3_m3 = reluConv(256, pool2_m3)

    conv3_m1 = reluConv(256, conv3_m1)
    conv3_m2 = reluConv(256, conv3_m2)
    conv3_m3 = reluConv(256, conv3_m3)

    vfr3_m1, vfr3_m2, vfr3_m3 = VFR([conv3_m1, conv3_m2, conv3_m3], 256)
    deconv3_m1 = concatenate([deConv(64, vfr3_m1), vfr2_m1], axis = 4)
    deconv3_m2 = concatenate([deConv(64, vfr3_m2), vfr2_m2], axis = 4)
    deconv3_m3 = concatenate([deConv(64, vfr3_m3), vfr2_m3], axis = 4)

    conv4_m1 = reluConv(128, deconv3_m1)
    conv4_m2 = reluConv(128, deconv3_m2)
    conv4_m3 = reluConv(128, deconv3_m3)

    conv4_m1 = reluConv(128, conv4_m1)
    conv4_m2 = reluConv(128, conv4_m2)
    conv4_m3 = reluConv(128, conv4_m3)

    vfr4_m1, vfr4_m2, vfr4_m3 = VFR([conv4_m1, conv4_m2, conv4_m3], 128)
    deconv4_m1 = concatenate([deConv(32, vfr4_m1), vfr1_m1], axis = 4)
    deconv4_m2 = concatenate([deConv(32, vfr4_m2), vfr1_m2], axis = 4)
    deconv4_m3 = concatenate([deConv(32, vfr4_m3), vfr1_m3], axis = 4)

    conv5_m1 = reluConv(64, deconv4_m1)
    conv5_m2 = reluConv(64, deconv4_m2)
    conv5_m3 = reluConv(64, deconv4_m3)

    conv5_m1 = reluConv(64, conv5_m1)
    conv5_m2 = reluConv(64, conv5_m2)
    conv5_m3 = reluConv(64, conv5_m3)

    vfr5 = concatenate(VFR([conv5_m1, conv5_m2, conv5_m3], 64), axis = 4)
    conv6 = reluConv(32, vfr5)
    out = softConv(1, conv6, name='OUT')

    model = Model(inputs = [inputs_m1, inputs_m2, inputs_m3], outputs = [out_ds1, out_ds2, out])
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000199), loss=soft_dice_loss, metrics=['accuracy', dice_coef])
    return model

def Unet2D():
    base_filt = 64
    inputs = Input((img_rows_trim, img_cols_trim, img_channels))
    conv1 = Conv2D(base_filt, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(base_filt, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*base_filt, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(2*base_filt, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4*base_filt, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(4*base_filt, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(8*base_filt, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(8*base_filt, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16*base_filt, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(16*base_filt, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(8*base_filt, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(8*base_filt, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(8*base_filt, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(4*base_filt, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(4*base_filt, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(4*base_filt, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(2*base_filt, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(2*base_filt, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(2*base_filt, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(base_filt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(base_filt, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(base_filt, (3, 3), activation='relu', padding='same')(conv9)

    # conv10 = Conv2D(5, (1, 1))(conv9)
    conv10 = Conv2D(5, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000199), loss=CCE, metrics=['accuracy', dice_coef])
 
    return model
