import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from codeUtils import *

def create_variable(name, shape=None, initializer=tf.keras.initializers.GlorotUniform(),
                    regularizer=None, train=True):
    return tf.compat.v1.get_variable(name=name, shape=shape, dtype=tf.float32, 
                initializer=initializer, regularizer=regularizer, trainable=train)

def l1(layer, sparsity=0.22):
    kernel, bias = layer.get_weights()
    n_f = kernel.shape[-1]
    zz = tf.concat([tf.reshape(tf.transpose(kernel, [3, 2, 0, 1]), [n_f, -1]), tf.reshape(bias, [n_f, 1])], 1)
    zz = tf.norm(zz, ord=1, axis=1)
    Lambda = sorted(zz)[int(n_f*sparsity)]
    return np.array([0 if x <Lambda else 1 for x in zz])

class PruneSSR:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.prune_list = []
        self.p = 1
        self.Lambda = 0.5

    def change_wts(self, layer, mkernel, _shape):
        k_h, k_w, k_c, n_f = _shape
        wt = mkernel[:, :-1]
        wt = tf.reshape(wt, [n_f, -1, k_h, k_w])
        wt = tf.transpose(wt, [2, 3, 1, 0])
        b = mkernel[:,-1]
        layer.set_weights([wt, b])

    def get_loss(self):
        # keras loss functions are not compatible use custom loss function
        return soft_dice_loss(self.data[1], self.model.predict(self.data[0], batch_size=1))
        # return CCE(self.data[1], self.model.predict(self.data[0], batch_size=1))

    def _do_ssr_layer(self, layer):

        kernel = layer.get_weights()[0]
        bias = layer.get_weights()[1]
        k_h, k_w, k_c, n_f = kernel.shape
        F = np.ones([n_f, k_c*k_h*k_w + 1])
        Y = np.zeros([n_f, k_c*k_h*k_w + 1])
        mkernel = tf.Variable(tf.concat([tf.reshape(tf.transpose(kernel, [3, 2, 0, 1]), [n_f, -1]), tf.reshape(bias, [n_f, 1])], 1))

        for it in range(5):
            T1 = F - 1 / self.p * Y
            loss = lambda : self.p/2 * tf.pow(tf.norm(mkernel - T1, ord='fro', axis=[0, 1]), 2)
            ''' 
            Here adding keras-CCE loss function is not compatible with keras-optimizer since every operation doeant have a gradient 
            so we use custom made loss function here 
            ''' 
            # loss = lambda : self.get_loss() + self.p/2 * tf.pow(tf.norm(mkernel - T1, ord='fro', axis=[0, 1]), 2)
            tf.keras.optimizers.SGD().minimize(loss, [mkernel])
            # set the model weight at this layer
            self.change_wts(layer, mkernel, kernel.shape)

            T2 = mkernel + 1 / self.p * Y
            # l2,1
            newF = T2 * tf.reshape(tf.maximum(tf.norm(T2, ord=2, axis=1) - self.Lambda / self.p, 0) / (tf.norm(T2, ord=2, axis=1) + 1e-9), [-1, 1])
            dF = newF-F
            F = newF

            Y = Y + self.p * (mkernel - F)
            if tf.norm(dF, ord=2)<=2:
                break

        return 1 - tf.cast(tf.equal(tf.reduce_sum(F, 1), 0), tf.float32)

    def _do_ssr(self):
        for layer in self.model.layers[:-1]:
            if layer.__class__.__name__ == 'Conv2D':
                #--------
                # mask has zero where a layer has to be pruned
                mask = self._do_ssr_layer(layer)
                # mask = l1(layer)
                #--------
                w_compress = 1 - tf.reduce_sum(mask) / mask.shape[0]
                channels = [i for i, x in enumerate(mask) if x==0.]
                self.prune_list.append((layer, channels))
            else:
                self.prune_list.append((layer, None))
        self.prune_list.append((self.model.layers[-1], None))

    def get_model(self):
        self.prune_list = []
        self._do_ssr()
        return delete(self.model, self.prune_list)

# delete channels in filter
def delete(model, lst):
    channels_to_remove = None
    newModelLayers = []
    convs = []
    last_deconv = None
    cc = 0
    for layer, indices in lst[:-1]:
        if layer.__class__.__name__ == 'Conv2DTranspose':
            last_deconv = (layer, []) # for now no pruning in deconv. later set to (layer, indices)
            wt, b = layer.get_weights()
            if not channels_to_remove is None:
                wt = np.delete(wt, channels_to_remove, axis=3)
                channels_to_remove=None
            newModelLayers.append((layer, (wt, b), wt.shape[-2]))
        elif layer.__class__.__name__ == 'Concatenate':
            if cc == 0:
                zz = 7
            elif cc == 1:
                zz = 5
            elif cc == 2:
                zz = 3
            elif cc == 3:
                zz = 1
            else:
                raise Exception('New concatenate found')
            cc +=1
            offset = last_deconv[0].output_shape[-1]
            assert channels_to_remove is None, "something is wrong"
            channels_to_remove = last_deconv[1]+[x+offset for x in convs[zz][1]]
            newModelLayers.append((layer, None, None))
        elif layer.__class__.__name__ == 'Conv2D':
            assert not indices is None, 'Pruning Indices for Conv2D is empty'
            convs.append((layer, indices))
            wt, b = layer.get_weights()
            if not channels_to_remove is None:
                wt = np.delete(wt, channels_to_remove, axis=2)
                channels_to_remove=None
            wt = np.delete(wt, indices, axis=-1)
            b = np.delete(b, indices, axis=-1)
            channels_to_remove = indices
            newModelLayers.append((layer, (wt, b), wt.shape[-1]))
        else:
            newModelLayers.append((layer, None, None))
    assert not channels_to_remove is None, "Last layer's channels should get pruned"
    wt, b = lst[-1][0].get_weights()
    wt = np.delete(wt, channels_to_remove, axis=2)
    newModelLayers.append((lst[-1][0], (wt, b), wt.shape[-1]))

    TEMP_conv2dT = None
    convs = []
    assn_wts = []
    TT = None
    cc = 0
    for layer, wts, n_f in newModelLayers[:-1]:
        if layer.__class__.__name__ == 'Conv2D':
            conv = Conv2D(n_f, (3, 3), activation='relu', padding='same')
            assn_wts.append((conv, wts))
            TT = conv(TT)
            convs.append(TT)
        elif layer.__class__.__name__ == 'InputLayer':
            temp = Input((144, 144, 4))
            INP = temp
            TT = temp
        elif layer.__class__.__name__ == 'MaxPooling2D':
            TT = MaxPooling2D(pool_size=(2, 2))(TT)
        elif layer.__class__.__name__ == 'Conv2DTranspose':
            deconv = Conv2DTranspose(n_f, (2, 2), strides=(2, 2), padding='same')
            TT = deconv(TT)
            assn_wts.append((deconv, wts))
            TEMP_conv2dT = TT
            convs.append(TT)
        elif layer.__class__.__name__ == 'Concatenate':
            if cc == 0:
                zz = 7
            elif cc == 1:
                zz = 5
            elif cc == 2:
                zz = 3
            elif cc == 3:
                zz = 1
            else:
                raise Exception('extra concatenate layer found')
            cc+=1
            TT = concatenate([TEMP_conv2dT, convs[zz]], axis=3)
        else:
            raise Exception(f'{layer.name}-Layer not found')

    OUT = Conv2D(5, (1, 1), activation='softmax')(TT)
    newmodel = Model(inputs=[INP], outputs=[OUT])
    for layer, wts in assn_wts:
        layer.set_weights(wts)
    newmodel.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000199), loss=CCE, metrics=['accuracy', dice_coef])
    return newmodel
