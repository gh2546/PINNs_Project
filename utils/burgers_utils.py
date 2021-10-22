## Importing required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from pyDOE import lhs
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

    usol-- a numpy array of shape (n_t, n_x) which information of the exact solution
    '''
    data = scipy.io.loadmat(file_path)
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    usol = np.real(data['usol']).T
    return x, t, usol

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

def test_data(X, T, usol):
    '''
    test_data: function returns the data where we will model will make predictions
    Arguments:
    X-- a numpy ndarray of shape (n_t, n_x); contains spatial coordinate
    matrices for x and t vectors

    T-- a numpy ndarray of shape (n_t, n_x); contains temporal coordinate
    matrices for x and t vectors

    usol-- a numpy array of shape (n_t, n_x) which information of the exact solution
    Returns:
    X_test-- a numpy ndarray of shape (n_t*n_x, 2); test data
    y_test-- a numpy ndarray of shape (n_t*n_x, 1); test labels
    '''
    X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    y_test = usol.flatten()[:,None]
    return X_test, y_test

def domain_bounds(X):
    '''
    domain_bounds: function returns the lower and the upper bound of the
    computational domain
    Arguments:
    X-- a numpy ndarray of shape (n_t*n_x, 2);
    Returns:
    lb, ub -- 2 numpy ndarray of shape (2,); contains spatial and temporal
    bound information
    '''
    lb = X.min(0)
    ub = X.max(0)
    return lb, ub

def training_data(X, T, usol, lb, ub, N_f):
    '''
    training_data: function returns the initial, boundary condition and
    collocation points data which is required during the training of the function
    Arguments:
    X-- a numpy ndarray of shape (n_t, n_x); contains spatial coordinate
    matrices for x and t vectors

    T-- a numpy ndarray of shape (n_t, n_x); contains temporal coordinate
    matrices for x and t vectors

    usol-- a numpy array of shape (n_t, n_x) which information of the exact solution

    b, ub-- 2 numpy ndarray of shape (2,); contains spatial and temporal
    bound information

    N_f-- number of collocation points
    Returns:
    x_f_train-- a numpy array of shape (n_x+2*n_t+N_f, 2); contains grid for f_train data
                column 1 corresponds to x and column2 corresponds to t;
                Boundary and Initial conditons + number of collocation points == (n_x + 2 * n_t + N_f)
    x_u_train-- a numpy array of shape (n_x + 2*n_t, 2); contains grid for u_train data
                column 1 corresponds to x and column2 corresponds to t
    y_u_train-- a numpy array of shape (n_x + 2*n_t, 1); contains label of the training data
    '''
    # xx_{} corresponds to the spatio-temporal grid for whatever written in {}
    # u_{} corresponds to the latent variables for whatever written in {}
    # IC ==> Initial condition
    # BC ==> Boundary condition

    # Initial conditions
    # xx_IC[:,0:1] represents x
    # xx_IC[:,1:2] represents t (t=0)
    xx_IC = np.hstack((X[0:1,:].T, T[0:1,:].T))
    # value of solution on the grid at t = 0
    uu_IC = usol[0:1,:].T

    # Boundary conditions (left)
    # xx_BC_left[:,0:1] represents (x = -1)
    # xx_BC_left[:,1:2] represents t
    xx_BC_left = np.hstack((X[:,0:1], T[:,0:1]))
    # value of the solution at x=-1 during the simulations (think about the temporal direction)
    uu_BC_left = usol[:,0:1]

    # Boundary conditions(right)
    # xx_BC_right[:,0:1] represents (x = 1)
    # xx_BC_right[:,1:2] represents t
    xx_BC_right = np.hstack((X[:,-1:], T[:,-1:]))
    # value of the solution at x=-1 during the simulations (think about the temporal direction)
    uu_BC_right = usol[:,-1:]

    # Creating spatial grid by stacking IC, BC vertically
    inp_u_x_train = np.vstack([xx_IC, xx_BC_left, xx_BC_right])

    # Creating spatial grid for collocation points
    # Generate a latin-hypercube design
    x_f_train = lb + (ub-lb)*lhs(2, N_f)

    # Solution of the latent variable for IC, BC conditoons
    y_u_train = np.vstack([uu_IC, uu_BC_left, uu_BC_right])

    # We are not returning y_f_train because it's value is always 0.
    return x_f_train, inp_u_x_train, y_u_train

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


@tf.function
def model_build_compile(x_f, t_f, x_u, t_u,u, shared_model, loss_function, optimizer):
    '''
    model_build_compile: function will build the model, compile with loss function and optimizer,
    calculate the gradients for weights and updating them automatically
    Arguemnts:
    x_f-- tensorflow batch dataset of shape (None, 1) [Pass x training data for collocation f]
    t_f-- tensorflow batch dataset of shape (None, 1) [Pass t training data for collocation f]
    x_u-- a numpy ndarray of shape (None, 1) [Pass x training data for latent variable u]
    t_u-- a numpy ndarray of shape (None, 1) [Pass t training data for latent variable u]
    u-- a numpy ndarray of shape (None, 1) [Pass training labels for latent variable u]
    shared_model-- the keras model which is shared between u and f network
    loss_function-- the keras loss function (for this project we use MSE)
    optimizer-- the keras optimizer function (please specify the name, learning rate)

    Returns
    loss_u-- a mean squared tensorflow loss calcuated for variable u
    loss_f-- a mean squared tensorflow loss calcuated for collocation f
    loss_combine-- combinted loss of variable u and collocation f loss_combine = loss_u + loss_f
    '''
    # We have 2 different loss functions here
    # loss_1 corresponds to loss in u
    # loss_2 corresponds to loss in f
    # Typicall plan, i.e. build model, compile model and fit model were not working in our case
    # We referred this link for training the model
    # Reference: https://www.tensorflow.org/guide/function
    # Reference: https://keras.io/api/optimizers/#learning-rate-decay--scheduling
    # Reference: https://colab.research.google.com/drive/1lo7Kf8zTb-DF_MjkO8Y07sYELnX3BNUR#scrollTo=GkimJNtepkKi
    # Reference: https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
    # Reference: https://github.com/helonayala/pinn/blob/main/01_DeepVIV_sec21.ipynb
    # This apprach i.e. optmizers.minimize(cost/loss) is also mentioned in the original paper

    # In the paper, they specify mean squared error for variable u and f

    # learning schedule for the optimizer

    # In the original paper, they used scipy "L-BFGS" optimizer but we are using Adam
    # L-BFGS is not available in tf. We can get it via tensorflow probability distribution but it's available
    # for tf>=2.3
#     if opt == 'Adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # for the pde structure we need some derivatives of u, so we are using gradient tape to do it for us
    # Reference: https://www.tensorflow.org/guide/autodiff
    with tf.GradientTape() as tape_out:
        # Trainign for variable u i.e IC and BC
        u_pred = shared_model([x_u, t_u], training=True)
        loss_u = loss_function(u, u_pred)

        # Trainign for variable f i.e collocation points
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(x_f)
            tape_in.watch(t_f)

            u_f = shared_model([x_f, t_f], training=True)
            # Making a PDE net using automatic differentiation technique
            # f = u_t + u * u_x - nu * u_xx
            # Calling gradient tape here, because want to trace everything
            # term1 ()
            u_x = tape_in.gradient(u_f, x_f)
            # term 2
            u_t = tape_in.gradient(u_f, t_f)
        #term 3
        u_xx = tape_in.gradient(u_x, x_f)

        f_pred = u_t + u_f * u_x - 0.01/np.pi*u_xx
        # Value of f_pred is zero
        loss_f = loss_function(f_pred, tf.convert_to_tensor(0, dtype=tf.float64))

        # Once we calculate loss for each variable and now combine them
        # MSE(u, u_pred) + MSE(0, f_pred)
        loss_combine = loss_u + loss_f

    # As given on the webpages
    grads = tape_out.gradient(loss_combine, shared_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, shared_model.trainable_weights))


    return loss_u, loss_f, loss_combine

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
