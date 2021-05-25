import tensorflow as tf
import os, time, datetime, math
import numpy as np
# from tensorflow.python import framework_ops
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from codeUtils import *
import Models
from pruningModels import prune_ssr

# project_name = '2D-Unet'
project_name = '2D-Unet-SSR'
# project_name = '3D-Unet'
# tf.compat.v1.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # switch off GPU

def scale(a):
    return (a-a.min())/(a.max()-a.min())

def trainLoop(model, X, y, batch_size=1, loss_fn=soft_dice_loss, optimizer=Adam(lr=0.001, decay=0.000199)):
    N = X.shape[0]
    max_steps = math.ceil(N/batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    epochs = 1
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)  # Logits for this minibatch
                loss_value = loss_fn(y_batch_train, logits)
                
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f"[{step}/{max_steps}]: {int(step*100/max_steps)}%  loss:{round(float(loss_value), 4)}", end='\r')

def train():
    print(f'\nProject Name: {project_name}')
    print('\n: Loading and preprocessing train data...\n')

    imgs, imgs_mask = get_data(200) if '3D' in project_name else get_data(dims='2D')
    # data is in float64
    imgs = scale(imgs) # scale to [0, 1]
    # imgs_mask = imgs_mask # scale to [0, 1]
    N = imgs.shape[0]
    z = int(0.8*N)
    imgs_train, imgs_val = tf.split(imgs, [z, N-z])
    imgs_mask_train, imgs_mask_val = tf.split(imgs_mask, [z, N-z])
    print('\n: Creating and compiling model...\n')

    model = Models.SWSM3D() if '3D' in project_name else Models.Unet2D()
    if 'SSR' in project_name:
        model.load_weights('weights/2D-Unet2021-05-18T18:33:29.847208.h5')
        model = prune_ssr(model)
    model.summary(line_length=130)

    weight_dir = 'weights'
    log_dir = 'logs'

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, f'{project_name}{datetime.datetime.now().isoformat()}.h5'), monitor='val_loss', save_best_only=True, mode='min')
    csv_logger = CSVLogger(os.path.join(log_dir,  f'{project_name}{datetime.datetime.now().isoformat()}.txt'), separator=',', append=False)

    print('\n: Fitting model...\n')

    # trainLoop(model, imgs_train, imgs_mask_train, batch_size=4)
    # return 

    start = time.time()
    if '3D' in project_name:
        model.fit(imgs_train[:,:,:,:,0:1], [imgs_mask_train]*3, batch_size=1, epochs=1, verbose=1, validation_split=0.2, callbacks=[model_checkpoint, csv_logger], use_multiprocessing=True)
    else:
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=40, verbose=2, validation_split=0.1, callbacks=[model_checkpoint, csv_logger], use_multiprocessing=True)
    
    print(time.time()-start)
    print('\n: Training finished, Now Testing\n')

    if '3D' in project_name:
        model.evaluate(imgs_val[:,:,:,:,0:1], [imgs_mask_val]*3, batch_size=1, verbose=1, callbacks=[model_checkpoint, csv_logger], use_multiprocessing=False)
    else:
        model.evaluate(imgs_val, imgs_mask_val, batch_size=1, verbose=1, callbacks=[model_checkpoint, csv_logger], use_multiprocessing=False)
    print(time.time()-start)
    if '3D' in project_name:
        a, b = get_input_data(dims='3D')
        a = scale(a)[...,0:1]
        out = model.predict(a, batch_size=1, verbose=1, callbacks=[model_checkpoint, csv_logger], use_multiprocessing=False)
        np.savez_compressed('X_inp.npz', a)
        np.savez_compressed('y_pred.npz', out)
        np.savez_compressed('y_true.npz', b)
    else:
        a, b = get_input_data(dims='2D')
        a = scale(a) # scal to [0, 1]
        # b = b.astype('float32')
        out = model.predict(a, batch_size=1, use_multiprocessing=False)
        np.savez_compressed('X_inp.npz', a)
        np.savez_compressed('y_pred_p.npz', out)
        np.savez_compressed('y_true.npz', b)

    print('\n: Testing finished')

if __name__ == '__main__':
    train()