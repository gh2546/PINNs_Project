## Importing required libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_loss(num_epoch, y_list, which):
    '''
    plot_loss: function plots the different losses in a same fig
    Arguments:
    num_epoch-- specify the number of epochs
    y_list-- a list which contains the list of tensorflow loss
    which-- specifies 'burgers' or 'schrodinger'
    Returns:
    None
    '''
    xlab = r"Number of epochs"
    ylab = r"Loss"
    a = min(num_epoch, len(y_list[0]))
    x = np.linspace(start = 0, stop = a,
            num = a, dtype = np.int)
    if which == 'burgers':
        labels = [r"$loss_u$", r"$loss_f$", r"$loss_{model}$"]
        color = ['k','r', 'g' ]
        i = 0
        plt.figure(figsize=(6,4))
        for loss in y_list:
            loss_num = list_tensor_to_list(loss)
            loss_num = list_tensor_to_list(loss)
            plt.plot(x, loss_num, color[i], label=labels[i])
            i += 1
        plt.yscale('log')
        plt.xlabel(xlab, fontsize=10)
        plt.ylabel(ylab, fontsize=10)
        plt.legend(loc='best',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.xlim([0, a])
        plt.savefig('../figures/Burgers/burgers_loss.png', dpi=500)
        plt.show()
    if which == 'schrod':
        # [loss_u_IC, loss_v_IC, loss_u_BC, loss_v_BC, loss_u_x_BC, loss_v_x_BC, loss_f_u, loss_f_v, loss_model]
        labels = [r"$u_{IC}$", r"$v_{IC}$", r"$u_{BC}$", r"$v_{BC}$", 
                  r"$u_{x_{BC}}$", r"$v_{x_{BC}}$", r"$f_{u}$", r"$f_{v}$",r"model"]
        i = 0
        plt.figure(figsize=(6,4))
        for loss in y_list:
            loss_num = list_tensor_to_list(loss)
            loss_num = list_tensor_to_list(loss)
            plt.plot(x, loss_num, label=labels[i])
            i += 1
        plt.yscale('log')
        plt.xlabel(xlab, fontsize=10)
        plt.ylabel(ylab, fontsize=10)
        plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.xlim([0, a])
        plt.savefig('../figures/Schrodinger/schrod_loss.png', dpi=500)
        plt.show()



def three_plots_burgers(x, y, y_hat):
    '''
    plot_loss: function plots the snapshots of the solution at different time
    Arguments:
    x-- spatial grid (256, 1)
    y-- ground truth solution
    y_hat-- predicted solution
    Returns:
    None
    '''
    plt.figure(figsize=(6, 4))
    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3,
                                    sharey='row')

    ax1.plot(x,y[25,:], 'b-', linewidth = 2, label = 'Exact')
    ax1.plot(x,y_hat[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$u(t,x)$')
    ax1.set_title('$t = 0.25$', fontsize = 10)
    ax1.axis('square')
    ax1.set_xlim([-1.1,1.1])
    ax1.set_ylim([-1.1,1.1])


    ax2.plot(x,y[50,:], 'b-', linewidth = 2, label = 'Exact')
    ax2.plot(x,y_hat[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$u(t,x)$')
    ax2.axis('square')
    ax2.set_xlim([-1.1,1.1])
    ax2.set_ylim([-1.1,1.1])
    ax2.set_title('$t = 0.50$', fontsize = 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax3.plot(x,y[75,:], 'b-', linewidth = 2, label = 'Exact')
    ax3.plot(x,y_hat[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$u(t,x)$')
    ax3.axis('square')
    ax3.set_xlim([-1.1,1.1])
    ax3.set_ylim([-1.1,1.1])
    ax3.set_title('$t = 0.75$', fontsize = 10)
    plt.savefig('../figures/Burgers/burgers_3.png', dpi=500)
    plt.tight_layout()
    plt.show()



def heat_map(U_pred, t, x, X_u_train, str):
    fig, ax = plt.subplots()
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    fig.savefig('../figures/Burgers/{}'.format(str))
    
def three_plots_schrod(x,t, Exact_h, H_pred):
    '''
    plot_loss: function plots the snapshots of the solution at different time
    Arguments:
    x-- spatial grid (256, 1)
    y-- ground truth solution
    y_hat-- predicted solution
    Returns:
    None
    '''
    plt.figure(figsize=(6, 4))
    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3,
                                    sharey='row')

    ax1.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax1.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$|h(t,x)|$')    
    ax1.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax1.axis('square')
    ax1.set_xlim([-5.1,5.1])
    ax1.set_ylim([-0.1,5.1])


    ax2.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax2.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$|h(t,x)|$')
    ax2.axis('square')
    ax2.set_xlim([-5.1,5.1])
    ax2.set_ylim([-0.1,5.1])
    ax2.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)


    ax3.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax3.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|h(t,x)|$')
    ax3.axis('square')
    ax3.set_xlim([-5.1,5.1])
    ax3.set_ylim([-0.1,5.1])    
    ax3.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    plt.savefig('../figures/Schrodinger/schrod_3.png', dpi=500)
    plt.tight_layout()
    plt.show()

def heat_map_schrod(H_pred, x,t,lb, ub, X_u_train, str1):
    '''
    Note: We took this section of the code directly from the original paper. It's just a plotting code and before taking this code, we asked Professor about this on piazza
    '''
    fig, ax = plt.subplots()
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    fig.savefig('../figures/Schrodinger/{}'.format(str1))
    return None
