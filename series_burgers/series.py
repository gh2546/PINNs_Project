import os
import numpy
import sys
sys.path.append("/home/ecbm4040/PINNS/utils")
from burgers_utils import *
from mlp_network import shared_model_func
from plotting import plot_loss, list_tensor_to_list, three_plots_burgers, heat_map

file_path = '/home/ecbm4040/PINNS/data/burgers_shock.mat'

# loading data from the burgers_shock.mat
# usol in the paper's script has been reffered as Exact
x, t, usol = load_data(file_path)

# making a spatial-temporal grid
X, T = mesh_grid(x, t)

# Creating test data set
# X_test is reffered as X_star
# y_test is reffered as u_star
X_test, y_test = test_data(X, T, usol)

# Domain bounds
lb, ub = domain_bounds(X_test)

# Number of collocation points i.e. number of training points for f
N_f = 10000
# Number of initial and boudary conditions to train the model for u
N_u = 100

# In the the paper they are reffered as X_u_train, X_f_train and u_train
x_f, x_u, y_u = training_data(X, T, usol, lb, ub, N_f)

# From the entire initial and boundary conditions we will select N_u number of x,t,u for the training
idx = np.random.choice(x_u.shape[0], N_u, replace=False)
X_u_train = x_u[idx, :]
y_u_train = y_u[idx, :]
# Stacking collocation points and IC and BC condtions
# this will be used for imposing the structure of the Partial Differential Equation
X_f_train = np.vstack([x_f, X_u_train])

layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 1]

# Specifying the coefficient of viscosity
nu = 0.01/np.pi

# Creating a shared neural network
shared_model = shared_model_func(layers=layers,
                                  lb=lb,
                                   ub = ub,
                                    norm = True)


# Specified in the paper
loss_function = tf.keras.losses.MeanSquaredError()
# Working perfect
# 1. choose_optimizer(lr=1e-02, ds=100, er=0.96, opt='Adam')
# 2. choose_optimizer(lr=1e-02, ds=100, er=0.96, opt='RMSprop')
optimizer = choose_optimizer(lr=1e-02, ds=100, er=0.96, opt='Adam', lear_rate_sched=True)
train_dataset = data_gen(X_f_train, batch_size=X_f_train.shape[0])
max_Iter = 10000
loss_u = []
loss_f = []
loss_model = []
loss_prev = 100000000000
for epoch in range(max_Iter):
    for (X_f_train, t_f_train) in train_dataset:
        # Passing the entire data in a single batch (That's how they did in the original paper)
        loss_1, loss_2, loss_combine = model_build_compile(x_f=X_f_train, t_f=t_f_train,
                                                           x_u = X_u_train[:,0:1], t_u = X_u_train[:,1:2],
                                                           u = y_u_train, shared_model=shared_model,
                                                           loss_function=loss_function,
                                                           optimizer=optimizer)
        loss_u.append(loss_1)
        loss_f.append(loss_2)
        loss_model.append(loss_combine)

        # if (epoch%500 == 0):
            # print("After {0} epochs loss on variable u is {1}".format(epoch, loss_1.numpy()))
            # print("After {0} epochs loss on PDE structure (f) is {1}".format(epoch, loss_2.numpy()))
            # print("After {0} epochs combined loss is {1}".format(epoch, loss_combine.numpy()))


y_hat = predict(shared_model, X_test)

loss = error_loss(y_test, y_hat)
print("*****************************************************\n")
print("Number of collocation points are: {0}\n".format(N_f))
print("Number of collocation points are: {0}\n".format(N_u))
print("Number of epochs are: {0}\n".format(max_Iter))
print("Network architecture is {0}\n".format(layers))
print("Number of Neurons in hidden layers are: {}\n".format(layers[2]))
print("Number of hidden layers in the network are: {}\n".format(len(layers)-2))
print("Error for the latent variable u is: {:e}\n".format(loss))
print("\n*****************************************************")
