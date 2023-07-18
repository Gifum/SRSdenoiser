# Loss function in keras
import tensorflow as tf
import tensorflow.keras.backend as K

def k_diff_axis_1(a):
    ''' 
    Aproximate the derivative of a tf tensor by finite differences
    '''  

    paddings = tf.constant([[0,0], [1,0],[0,0]])
    a=tf.pad(a,paddings,"CONSTANT")
    return a[:,1:,:]-a[:,:-1,:]


def loss_def_norm1_modifiedL1(Grad_weight=.6,Weight=5.):
    ''' 
    Define the custom loss function as the sum of a l2 reconstruction term and a gradient l1 term
    Outputs:
        custom_mse: weighted sum of grad_loss and mse_loss
        grad_loss: l1 loss of the gradients
        mse_loss: l2 loss

    Inputs:
        Grad_weight: relative weight of the gradient term with respect to the reconstruction term (must be between 0 and 1)
        Weight: offset factor fixed once per training set to regularize abrupt changes in the loss decay during training when grad loss is switched on
    '''    

    def custom_mse(y_true, y_pred):

        # calculating squared difference between target and predicted values 
        loss2 = K.square(y_pred - y_true)/(K.square(y_true+1))   # (batch_size, 2)
        D_pred = k_diff_axis_1(y_pred)
        D_true = k_diff_axis_1(y_true)
        loss_grad = K.abs(D_pred - D_true)/K.abs(D_true+1)



        loss=(1-Grad_weight) * loss2 + Grad_weight * Weight * loss_grad



        return loss

    def grad_loss(y_true, y_pred):

        # calculating squared difference between target and predicted values 
        D_pred = k_diff_axis_1(y_pred)
        D_true = k_diff_axis_1(y_true)
        loss_grad = K.abs(D_pred - D_true)/K.abs(D_true+1)

        loss=Weight*loss_grad

        return loss

    def mse_loss(y_true, y_pred):

        loss = K.square(y_pred - y_true) /(K.square(y_true+1))

        return loss
    
    return custom_mse,grad_loss,mse_loss





def loss_def_norm1(Grad_weight=.6,Weight=5.):

    ''' 
    Define the custom loss function as the sum of a l2 reconstruction term and a gradient l2 term
    Outputs:
        custom_mse: weighted sum of grad_loss and mse_loss
        grad_loss: l2 loss of the gradients
        mse_loss: l2 loss

    Inputs:
        Grad_weight: relative weight of the gradient term with respect to the reconstruction term (must be between 0 and 1)
        Weight: offset factor fixed once per training set to regularize abrupt changes in the loss decay during training when grad loss is switched on
    '''

    def custom_mse(y_true, y_pred):

        # calculating squared difference between target and predicted values 
        loss2 = K.square(y_pred - y_true)/(K.square(y_true+1))   
        D_pred = k_diff_axis_1(y_pred)
        D_true = k_diff_axis_1(y_true)
        loss_grad = K.square(D_pred - D_true) /K.square(D_true+1)
  
        loss=(1-Grad_weight) * loss2 + Grad_weight * Weight * loss_grad



        return loss

    def grad_loss(y_true, y_pred):

        # calculating squared difference between target and predicted values 
        D_pred = k_diff_axis_1(y_pred)
        D_true = k_diff_axis_1(y_true)
        loss_grad = K.square(D_pred - D_true) /K.square(D_true+1)

        loss=Weight*loss_grad

        return loss

    def mse_loss(y_true, y_pred):

        loss = K.square(y_pred - y_true) /(K.square(y_true+1))

        return loss
    
    return custom_mse,grad_loss,mse_loss