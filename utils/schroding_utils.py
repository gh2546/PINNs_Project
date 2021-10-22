## Importing required libraries
import os
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

def load_data(file_path):
    '''
    load_data: function loads the data into the python from the "*.mat" files
    Arguments:
    file_path-- provide a path of the file which you want to load (NOTE: must be a string)
    Return:
    x-- a numpy ndarray of shape (n_x,1) which contains information of the
    spatial discretization

    t-- a numpy ndarray of shape (n_t,1) which contains information of the
    temporal discretization

    usol-- a numpy array of shape (n_x, n_t) which contains information of the exact solution

    usol_real-- a numpy array of shape (n_x, n_t) which contains information of
                real part of the exact solution

    usol_img-- a numpy array of shape (n_x, n_t) which contains information of
                imaginart part of the exact solution

    usol_mag-- a numpy array of shape (n_x, n_t) which contains information of
                real part of the exact solution
    '''
    data = scipy.io.loadmat(file_path)
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    usol = data['uu']
    usol_real = np.real(usol)
    usol_img = np.imag(usol)
    usol_mag = np.sqrt(usol_img**2 + usol_real**2 )

    assert usol_img.shape == usol_real.shape
    assert usol.shape == usol_mag.shape
    assert usol.shape[0] == x.shape[0]
    assert usol.shape[1] == t.shape[0]
    return x, t, usol, usol_real, usol_img, usol_mag

def mesh_grid(x, t):
    '''
    mesh_grid: function creat a spatio-temporal grid
    Arguments:
    x-- a numpy ndarray of shape (n_x,1) which contains information of the
    spatial discretization

    t-- a numpy ndarray of shape (n_t,1) which contains information of the
    temporal discretization

    Return:
    X-- a numpy ndarray of shape (n_t, n_x); contains spatial coordinate
    matrices for x and t vectors

    T-- a numpy ndarray of shape (n_t, n_x); contains temporal coordinate
    matrices for x and t vectors
    '''
    X, T = np.meshgrid(x,t)
    return X, T

def test_data(X, T, usol_real, usol_img, usol_mag):
    '''
    test_data: function returns the data where we will model will make predictions
    Arguments:
    X-- a numpy ndarray of shape (n_t, n_x); contains spatial coordinate
    matrices for x and t vectors

    T-- a numpy ndarray of shape (n_t, n_x); contains temporal coordinate
    matrices for x and t vectors

    usol_real, usol_img, usol_mag -- a numpy array of shape (n_x, n_t) which
     information of the real, imaginry and the magnitude of the exact solution
    Returns:
    X_test-- a numpy ndarray of shape (n_t*n_x, 2); test data
    y_u_test-- a numpy ndarray of shape (n_t*n_x, 1); test labels for real part
    y_v_test-- a numpy ndarray of shape (n_t*n_x, 1); test labels for imaginar part
    y_h_test-- a numpy ndarray of shape (n_t*n_x, 1); test labels for magnitude
    '''
    X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    y_u_test = usol_real.T.flatten()[:,None]
    y_v_test = usol_img.T.flatten()[:,None]
    y_h_test = usol_mag.T.flatten()[:,None]

    return X_test, y_u_test, y_v_test, y_h_test

def domain_bounds(x_min, x_max, t_min, t_max):
    '''
    domain_bounds: function returns the lower and the upper bound of the
    computational domain
    Arguments:
    x_min, x_max, t_min, t_max-- minimum and maximum value of space and time
    Returns:
    lb, ub -- 2 numpy ndarray of shape (2,); contains spatial and temporal
    bound information
    '''
    assert int(t_min) == 0
    lb = np.array([x_min, t_min])
    ub = np.array([x_max, t_max])
    return lb, ub

def data_gen(X_train, batch_size):
    '''
    data_gen: function converts the data for f points into the specified batch sizes
    Arguemnts:
    X_train-- a numpy array of size (n_train, 2)
    batch_size-- number o examples you want in the batches
    Returns:
    train_dataset-- a tensorflow batch data set consist of 2 variables x and t.
    '''
    assert X_train.shape[1] == 2
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train[:,0:1], X_train[:,1:2]))
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset

def predict(model, X_test):
    '''
    predict: function makes the prediction for the input data using trained model
    Arguments:
    model-- a trained keras model
    X_test-- a numpy ndarray of shape (None, 2);
    Returns:
    y_hat-- a numpy ndarray of shape (None, 1); prediction made by trained model
    '''
    assert X_test.shape[1] == 2
    y_hat = model.predict([X_test[:,0:1], X_test[:,1:2]])
    return y_hat

def error(y1,y2):
    '''
    '''
    assert y1.shape == y2.shape
    y = y1-y2
    return np.square(y).sum()**0.5

def list_tensor_to_list(list_tens):
    '''
    list_tensor_to_list: function convets the elements of list from tensor to numpy
    Arguments:
    list_tens-- a list type object, whose elements are tensorflow
    Returns:
    list_norm-- a list type object, with elements as numpy
    '''
    list_norm = []
    for l in list_tens:
        list_norm.append(l.numpy())
    return list_norm

def choose_optimizer(lr=1e-02, ds=100, er=0.96, opt='Adam', lear_rate_sched=True):
    '''
    choose_optimizer: function will choose the optimizer for you.
    Arguments:
    lr-- specify the learning rate; default value is 1e-02
    ds--  specify the decay steps for learning rate; default value is 100
    er-- specify the decay rate of learning rate; default value is 0.96
    opt-- specify the name for optimizer; default set is 'Adam';
            Options available are: 'Adam', 'SGD', 'Nadam', 'RMSprop'
    lear_rate_sched-- To use the ExponentialDecay schedules for the learning rate;
                    default value is True
                    if True then we will use the ExponentialDecay learning rate
                    else constant value of learning rate
    '''
    if lear_rate_sched == True:
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                        initial_learning_rate=lr,
                                                        decay_steps=ds,
                                                        decay_rate=er)
    else:
        learning_rate_schedule = lr


    if opt == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule,
                                            beta_1=0.9, beta_2=0.999,
                                            epsilon=1e-07, amsgrad=False,
                                            name='Adam')
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule,
                                           momentum=0.0, nesterov=False,
                                           name='SGD')
    if opt == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9,
                                            beta_2=0.999, epsilon=1e-07,
                                            name='Nadam')
    if opt == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule,
                                                rho=0.9, momentum=0.0, epsilon=1e-07,
                                                centered=False, name='RMSprop')
    return optimizer

def error_loss(y, y_hat):
    '''
    error_loss: function return error between predicted and actual value
    Arguments:
    y-- actual value
    y_hat-- predicted value
    Returns:
    error-- a scalar
    '''
    error = np.linalg.norm(y-y_hat, 2)/np.linalg.norm(y,2)
    return error

@tf.function
def model_build_compile(x0, t0, x_lb, t_lb, x_ub, t_ub, x_f, t_f,shared_model, loss_function, optimizer, u0, v0):
    '''
    '''
    # Typicall plan, i.e. build model, compile model and fit model were not working in our case
    # We referred this link for training the model
    # Reference: https://www.tensorflow.org/guide/function
    # Reference: https://keras.io/api/optimizers/#learning-rate-decay--scheduling
    # Reference: https://colab.research.google.com/drive/1lo7Kf8zTb-DF_MjkO8Y07sYELnX3BNUR#scrollTo=GkimJNtepkKi
    # Reference: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
    # Reference: https://github.com/helonayala/pinn/blob/main/01_DeepVIV_sec21.ipynb
    # This apprach i.e. optmizers.minimize(cost/loss) is also mentioned in the original paper
    # Reference: https://www.tensorflow.org/guide/autodiff
    with tf.GradientTape() as tape_out:
        uv = shared_model([x0, t0], training=True)
        u0_pred = uv[:,0:1]
        v0_pred = uv[:,1:2]

        ## Loss function for initial conditon
        loss_1 = loss_function(u0_pred, u0) # Real part
        loss_2 = loss_function(v0_pred, v0) # Imaginary part

        with tf.GradientTape(persistent=True) as tape_1:
            tape_1.watch(x_lb)
            uv_lb = shared_model([x_lb, t_lb], training=True)
            u_lb_pred = uv_lb[:,0:1]
            v_lb_pred = uv_lb[:,1:2]
            u_x_lb_pred = tape_1.gradient(u_lb_pred, x_lb)
            v_x_lb_pred = tape_1.gradient(v_lb_pred, x_lb)


        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(x_ub)
            uv_ub = shared_model([x_ub, t_ub], training=True)
            u_ub_pred = uv_ub[:,0:1]
            v_ub_pred = uv_ub[:,1:2]
            u_x_ub_pred = tape_2.gradient(u_ub_pred, x_ub)
            v_x_ub_pred = tape_2.gradient(v_ub_pred, x_ub)

        ## Loss functions for the boundary conditions
        loss_3 = loss_function(u_lb_pred, u_ub_pred)
        loss_4 = loss_function(v_lb_pred, v_ub_pred)
        loss_5 = loss_function(u_x_lb_pred, u_x_ub_pred)
        loss_6 = loss_function(v_x_lb_pred, v_x_ub_pred)



        with tf.GradientTape(persistent=True) as tape_3:
            tape_3.watch(x_f)
            tape_3.watch(t_f)
            uv_f = shared_model([x_f, t_f], training=True)
            u_f = uv_f[:,0:1]
            v_f = uv_f[:,1:2]
            u_x_f = tape_3.gradient(u_f, x_f)
            v_x_f = tape_3.gradient(v_f, x_f)
            u_t_f = tape_3.gradient(u_f, t_f)
            v_t_f = tape_3.gradient(v_f, t_f)
        u_xx_f = tape_3.gradient(u_x_f, x_f)
        v_xx_f = tape_3.gradient(v_x_f, x_f)

        f_u_pred = u_t_f + 0.5 * v_xx_f + (u_f**2 + v_f**2)*v_f
        f_v_pred = v_t_f - 0.5 * u_xx_f - (u_f**2 + v_f**2)*u_f

        ## Loss function for the collocation points
        loss_7 = loss_function(f_u_pred, tf.convert_to_tensor(0, dtype=tf.float64))
        loss_8 = loss_function(f_v_pred, tf.convert_to_tensor(0, dtype=tf.float64))

        loss_combine = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8
    grads = tape_out.gradient(loss_combine, shared_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, shared_model.trainable_weights))

    return loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8, loss_combine

