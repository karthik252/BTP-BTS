import os
import numpy as np
import nibabel as nib
import PIL
import tensorflow as tf
import multiprocessing, time
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

PROCESSED_DIR = '../ProcessedData/Train'
SAVED_DATA = '../ProcessedData/data.npz'
SAVED_DATA_2D = '../ProcessedData/data2d.npz'
MODS = ['t1_', 't1c_', 't2_', 'flair_', 'seg.'] # Dont chage the order

img_rows = 240
img_rows_trim = 144
img_cols = 240
img_cols_trim = 144
img_depth = 160
img_depth_trim = 80
img_channels = 4

IN_SIZE = 240
IN_SIZE_TRIM = 144
IN_DEPTH = 155
IN_CHANNELS = 4
OUT_CHANNEL = 1

# Metric
def dice_coef(y_true, y_pred):
    smooth = 1e-6 # check the correct value
    y_pred_1 = y_pred
    y_true_1 = tf.one_hot(tf.cast(y_true[...,0], tf.int64), depth=5, axis=-1)
    num = K.sum(y_true_1 * y_pred_1, axis=(0, 1, 2))
    den = K.sum(y_true_1, axis=(0, 1, 2)) + K.sum(y_pred_1, axis=(0, 1, 2))

    dice = K.mean((2. * num + smooth)/(den + smooth))
    return dice

# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# Loss fn
# shapes (y_true, y_pred) = ([batch, 144, 144, 80, 1], [batch, 144, 144, 80, 5{onehot}])
def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def CCE(y_true, y_pred):
    y_pred_1 = y_pred
    y_true_1 = tf.one_hot(tf.cast(y_true[...,0], tf.int64), depth=5, axis=-1)
    return tf.keras.losses.CategoricalCrossentropy()(y_true_1, y_pred_1)

# Loss fn
def weighted_CCE(weights): # weights might be constant or dependent on inputs class volume
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        y_true = tf.one_hot(tf.cast(y_true, tf.int64), 5)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Data handler
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def get_pat(param):
    out = None
    ans = []
    for ss in MODS:
        for mod in os.listdir(os.path.join(param[0], param[1])):
            if ('_'+ss) in mod:
                ans.append(np.expand_dims(np.array(nib.load(os.path.join(param[0], param[1], mod)).dataobj), axis=-1))

    # padding Depth axis
    ans = np.pad(np.concatenate(ans, axis=-1), ((0,0),(0,0),(2,3),(0,0)),'constant',constant_values=(0,0))
    return np.expand_dims(ans, axis=0)

def get_data(LIMIT=None, dims='3D'):
    N = len(os.listdir(PROCESSED_DIR))
    DIR = SAVED_DATA_2D if dims == '2D' else SAVED_DATA
    if os.path.exists(DIR):
        print('Loading saved data...')
        saved_data = np.load(DIR)
        if LIMIT is None:
            return saved_data['arr_0'], saved_data['arr_1']
        else:
            _a = max(1, min(LIMIT, N-1))
            return saved_data['arr_0'][:_a], saved_data['arr_1'][:_a]

    start = time.time()
    with multiprocessing.Pool() as p:
        DATA = np.concatenate(p.map(get_pat, [(PROCESSED_DIR, x) for x in os.listdir(PROCESSED_DIR)]), axis=0)
    print(time.time() - start)

    print('DATA collected')
    print('DATA.shape:', DATA.shape)
    # Cropping DATA
    DATA = DATA[:, 48:192, 48:192, 30:110, :]
    if dims == '2D':
        N *= 80
        print("flattening to 2D images")
        DATA = np.concatenate([DATA[:,:,:,i,:] for i in range(80)], axis = 0)
    np.random.shuffle(DATA)
    if dims == '2D':
        DATA_X, DATA_y = DATA[:,:,:,:4], DATA[:,:,:,4:]
        np.savez_compressed(SAVED_DATA_2D, DATA_X, DATA_y)
    else:
        DATA_X, DATA_y = DATA[:,:,:,:,:4], DATA[:,:,:,:,4:]
        np.savez_compressed(SAVED_DATA, DATA_X, DATA_y)

    # return Shape for 3D [:, 144, 144, 80, 4], [:, 144, 144, 80, 1]
    # return Shape for 2D [:, 144, 144, 4], [:, 144, 144, 1]
    if LIMIT is None:
        return DATA_X, DATA_y
    else:
        _a = max(1, min(LIMIT, N-1))
        return DATA_X[:_a], DATA_y[:_a]

def get_input_data(path=None, dims='3D'):
    assert  path is None, "get_input_data function is incomplete"
    data = get_pat(( PROCESSED_DIR, os.listdir(PROCESSED_DIR)[0] ))
    data = data[:, 48:192, 48:192, 30:110, :]
    if dims == '2D':
        data = np.concatenate([data[:,:,:,i,:] for i in range(80)], axis = 0)
        return data[:,:,:,:4], data[:,:,:,4:]
    else:
        return data[:,:,:,:,:4], data[:,:,:,:,4:]