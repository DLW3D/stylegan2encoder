import os
import numpy as np
import cv2

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Reshape
from keras.models import Sequential, Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam

import pretrained_networks
import dnnlib.tflib as tflib


def get_batch(batch_size, Gs, image_size=224, Gs_minibatch_size=12, w_mix=None):
    """
    Generate a batch of size n for the model to train
    returns a tuple (W, X) with W.shape = [batch_size, 18, 512] and X.shape = [batch_size, image_size, image_size, 3]
    If w_mix is not None, W = w_mix * W0 + (1 - w_mix) * W1 with
        - W0 generated from Z0 such that W0[:,i] = constant
        - W1 generated from Z1 such that W1[:,i] != constant

    Parametersget_batch
    ----------
    batch_size : int
        batch size
    Gs
        StyleGan2 generator
    image_size : int
    Gs_minibatch_size : int
        batch size for the generator
    w_mix : float

    Returns
    -------
    tuple
        dlatent W, images X
    """

    # Generate W0 from Z0
    Z0 = np.random.randn(batch_size, Gs.input_shape[1])
    W0 = Gs.components.mapping.run(Z0, None, minibatch_size=Gs_minibatch_size)

    if w_mix is None:
        W = W0
    else:
        # Generate W1 from Z1
        Z1 = np.random.randn(18 * batch_size, Gs.input_shape[1])
        W1 = Gs.components.mapping.run(Z1, None, minibatch_size=Gs_minibatch_size)
        W1 = np.array([W1[batch_size * i:batch_size * (i + 1), i] for i in range(18)]).transpose((1, 0, 2))

        # Mix styles between W0 and W1
        W = w_mix * W0 + (1 - w_mix) * W1

    # Generate X
    X = Gs.components.synthesis.run(W, randomize_noise=True, minibatch_size=Gs_minibatch_size, print_progress=True,
                                    output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

    # Preprocess images X for the Imagenet model
    X = np.array([cv2.resize(x, (image_size, image_size)) for x in X])
    X = preprocess_input(X.astype('float'))

    return W, X


def finetune(save_path, image_size=224, base_model=ResNet50, batch_size=2048, test_size=1024, n_epochs=6,
             max_patience=5):
    """
    Finetunes a ResNet50 to predict W[:, 0]

    Parameters
    ----------
    save_path : str
        path where to save the Resnet
    image_size : int
    base_model : keras model
    batch_size :  int
    test_size : int
    n_epochs : int
    max_patience : int

    Returns
    -------
    None

    """

    assert image_size >= 224

    # Load StyleGan generator
    _, _, Gs = pretrained_networks.load_networks('models/stylegan2-ffhq-config-f.pkl')

    # Build model
    if os.path.exists(save_path):
        print('Loading pretrained network')
        model = load_model(save_path, compile=False)
    else:
        base = base_model(include_top=False, pooling='avg', input_shape=(image_size, image_size, 3))
        model = Sequential()
        model.add(base)
        model.add(Dense(512))

    model.compile(loss='mse', metrics=[], optimizer=Adam(3e-4))
    model.summary()

    # Create a test set
    print('Creating test set')
    W_test, X_test = get_batch(test_size, Gs)

    # Iterate on batches of size batch_size
    print('Training model')
    patience = 0
    best_loss = np.inf

    while (patience <= max_patience):
        W_train, X_train = get_batch(batch_size, Gs)
        model.fit(X_train, W_train[:, 0], epochs=n_epochs, verbose=True)
        loss = model.evaluate(X_test, W_test[:, 0])
        if loss < best_loss:
            print(f'New best test loss : {loss:.5f}')
            model.save(save_path)
            patience = 0
            best_loss = loss
        else:
            print(f'-------- test loss : {loss:.5f}')
            patience += 1


def finetune_18(save_path, base_model=None, image_size=224, batch_size=2048, test_size=1024, n_epochs=6,
                max_patience=8, w_mix=0.7):
    """
    Finetunes a ResNet50 to predict W[:, :]

    Parameters
    ----------
    save_path : str
        path where to save the Resnet
    image_size : int
    base_model : str
        path to the first finetuned ResNet50
    batch_size :  int
    test_size : int
    n_epochs : int
    max_patience : int
    w_mix : float

    Returns
    -------
    None

    """

    assert image_size >= 224
    if not os.path.exists(save_path):
        assert base_model is not None

    # Load StyleGan generator
    _, _, Gs = pretrained_networks.load_networks('models/stylegan2-ffhq-config-f.pkl')

    # Build model
    if os.path.exists(save_path):
        print('Loading pretrained network')
        model = load_model(save_path, compile=False)
    else:
        base_model = load_model(base_model)
        hidden = Dense(18 * 512)(base_model.layers[-1].input)
        outputs = Reshape((18, 512))(hidden)
        model = Model(base_model.input, outputs)
        # Set initialize layer
        W, b = base_model.layers[-1].get_weights()
        model.layers[-2].set_weights([np.hstack([W] * 18), np.hstack([b] * 18)])

    model.compile(loss='mse', metrics=[], optimizer=Adam(1e-4))
    model.summary()

    # Create a test set
    print('Creating test set')
    W_test, X_test = get_batch(test_size, Gs, w_mix=w_mix)

    # Iterate on batches of size batch_size
    print('Training model')
    patience = 0
    best_loss = np.inf

    while (patience <= max_patience):
        W_train, X_train = get_batch(batch_size, Gs, w_mix=w_mix)
        model.fit(X_train, W_train, epochs=n_epochs, verbose=True)
        loss = model.evaluate(X_test, W_test)
        if loss < best_loss:
            print(f'New best test loss : {loss:.5f}')
            model.save(save_path)
            patience = 0
            best_loss = loss
        else:
            print(f'-------- test loss : {loss:.5f}')
            patience += 1


if __name__ == '__main__':
    finetune('data/resnet.h5')
    finetune_18('data/resnet_18.h5', 'data/resnet.h5', w_mix=0.8)
