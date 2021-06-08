from matplotlib import pyplot as plt
import numpy as np
import sys

def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps

def prettyplot(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
    plt.xlabel(xlabel, labelpad = xlabelpad)
    plt.ylabel(ylabel, labelpad = ylabelpad)

'''
def high_fidelity_boundary(X):
    return (0.48 + np.sin(2.5 * X[:, 0] * np.pi) / 6)

def high_fidelity_boundary_bis(X):
    return (0.5 - np.sin(2.2 * X[:, 0] * np.pi) / 12)
'''

def plot_multiclass(pred, pred_bis, h, X, Y, boundary, boundary_bis,filename='multiclass.png'):
    assert  X.shape[1] == 2, 'can only plot 2D functions'

    #compute complexive probability (to determine class)
    prob_cand = invlogit(pred).mean(0)
    ent_cand = -np.abs(pred.mean(0)) / (pred.std(0) + 1e-9)


    prob_cand_bis = invlogit(pred_bis).mean(0)
    ent_cand_bis = -np.abs(pred_bis.mean(0)) / (pred_bis.std(0) + 1e-9)

    #contour plot of classes
    plt.figure(1, figsize=(5.5, 10))
    plt.clf()
    plt.subplot(211)
    xx, yy = np.meshgrid(np.arange(0, 1 + h, h), np.arange(0, 1 + h, h))
    prob = (prob_cand + 2 * prob_cand_bis)/4
    plt.contourf(xx, yy, np.reshape(prob, xx.shape))
    cb = plt.colorbar(ticks=[0, 1])
    cb.set_label('class probability [-]', labelpad=-10)
    #plt.scatter(xx, yy, c=np.reshape(prob, xx.shape))

    labels = cb.ax.get_yticklabels()
    labels[0].set_verticalalignment("bottom")
    labels[-1].set_verticalalignment("top")

    #plot high fidelity points
    plt.scatter(X[:-1, 0], X[:-1, 1], c=Y[:-1])

    #plot boundaries
    x = np.linspace(0, 1, 100)[:, None]

    plt.plot(x, boundary(x), 'k--', label='high-fidelity boundary')
    plt.plot(boundary_bis(x), x, 'k--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])


    prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad=-10)


    #plot active learning
    '''  
    plt.subplot(212)
    ent = (ent_cand +  ent_cand_bis) / 2
    plt.contourf(xx, yy, np.reshape(ent, xx.shape))
    plt.colorbar(label = 'active learning [-]', ticks = [])
    plt.scatter(X[:-1, 0], X[:-1, 1])

    prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad=-10)
    '''
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


