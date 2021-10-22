import tensorflow as tf
def shared_model_func(layers, lb=None, ub=None, norm=False):
    '''
    shared_model_func: function creates a multilayer perceptron network for the shared network (u and f)
    Arguments:
    layers-- a list which contains of units in each layer
    Returns:
    model-- return a untrained keras model
    '''
    # u and f are function of x and t so creating 2 variables
    x =  tf.keras.Input(shape=[1,], dtype=tf.float64)
    t =  tf.keras.Input(shape=[1,], dtype=tf.float64)

    # Concating x and t so we can make an array of [x,t]
    inputs = tf.keras.layers.concatenate([x, t],dtype=tf.float64)
    # If want to standardize our data
    if norm==True:
        # Data pre-processing
        # Standardizing the data
        inputs = 2.0 * (inputs - lb)/(ub - lb) - 1.0
    # Xavier initializer and seed was not paper in the original paper but we set it as 1234
    # In case, TA wants to reproduce the results so they can do it
    initializer = tf.keras.initializers.glorot_normal(seed = 1234)
    for layer in layers[1:-1]:
        inputs = tf.keras.layers.Dense(units=int(layer),
                                       activation='tanh',
                                       kernel_initializer=initializer,
                                       dtype=tf.float64,
                                      trainable=True)(inputs)

    outputs = tf.keras.layers.Dense(units=int(layers[-1]),
                                    activation='linear',
                                    kernel_initializer=initializer,
                                    dtype=tf.float64,
                                   trainable=True)(inputs)

    model = tf.keras.models.Model(inputs=[x,t], outputs=outputs)
    return model
