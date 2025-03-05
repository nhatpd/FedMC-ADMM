import numpy as np

class Exp_Norm(object):
    def __init__(self, coeff=1.):
        self.coeff = coeff

    def func_eval(self, x):
        """! Compute the function value of the \f$\ell_1\f$ - norm
        
        Parameters
        ---------- 
        @param x : input vector
            
        Returns
        ---------- 
        @retval : function value
        """
        theta = 5
        return self.coeff*np.sum(1-np.exp(-theta*np.abs(x)))

    def prox_eval(self, x, prox_param, x_model = [0,0]):
        """! Compute the proximal operator of the \f$\ell_1\f$ - norm

        \f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
        
        Parameters
        ---------- 
        @param w : input vector
        @param prox_param : penalty paramemeter
            
        Returns
        ---------- 
        @retval : output vector
        """
        theta = 5
        if x_model[1] == 'fedADMM':
            w = theta*np.exp(-theta*np.abs(x_model[0]))
            prox_param *= self.coeff*w
            return np.sign( x ) * np.maximum( np.abs( x ) - prox_param, 0 )
        else:
            y = x_model[0]
            for iter in range(20):

                w = theta*np.exp(-theta*np.abs(y))
                prox_param *= self.coeff*w
                y = np.sign( x ) * np.maximum( np.abs( x ) - prox_param, 0 )
            return y
