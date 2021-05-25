import os, time
import tensorflow as tf
import numpy as np
from codeUtils import *
from Models import Unet2D
from pruningModels import prune_ssr
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # enable or disable GPU
best_model_2D = 'weights/2D-Unet2021-05-23T15:54:09.719219.h5'
best_model_ssr = 'weights/2D-Unet-SSR2021-05-23T21:09:36.275754.h5'
def scale(a):
    return (a-a.min())/(a.max()-a.min())

model = Unet2D()
model.load_weights(best_model_2D)

pruned_model = prune_ssr(model)

a, b = get_data(dims='2D')
a = scale(a)[:8000] # scal to [0, 1]
# b = b.astype('float32')
# start = time.time()
# out = model.predict(a, batch_size=1, use_multiprocessing=False)
# print(time.time()-start)
start = time.time()
out = pruned_model.predict(a, batch_size=1, use_multiprocessing=False)
print(time.time()-start)
# np.savez_compressed('X_inp.npz', a)
# np.savez_compressed('y_pred.npz', out)
# np.savez_compressed('y_true.npz', b)