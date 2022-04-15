import numpy as np

def conv_forward_naive(x, w, b):

    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape
    # print(x.shape)
    # print(w.shape)
   
    outH = 1 + (H - FH)
    outW = 1 + (W - FW )

    # create output tensor after convolution layer
    out = np.zeros((N, F, outH, outW))

    # create w_row matrix
    w_row = w.reshape(F, C*FH*FW)                            
    x_col = np.zeros((C*FH*FW, outH*outW))                   #[C*FH*FW x H'*W']
    for index in range(N):
        # print(index)
        neuron = 0 
        for i in range(0, outH):
            for j in range(0, outW):
                x_col[:,neuron] = x[index,:,i:i+FH,j:j+FW].reshape(C*FH*FW)
                neuron += 1
        out[index] = (w_row.dot(x_col) + b.reshape(F,1)).reshape(F, outH, outW)
 
    cache = (out, x, w, b)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    
    out, x, w, b = cache
    N, F, outH, outW = dout.shape
    N, C, H, W = x.shape
    FH, FW = w.shape[2], w.shape[3]
 

    # initialize gradients
    dx = np.zeros((N, C, H , W))
    dw, db = np.zeros(w.shape), np.zeros(b.shape)

    # create w_row matrix
    w_row = w.reshape(F, C*FH*FW)                            #[F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*FH*FW, outH*outW))                   #[C*FH*FW x H'*W']

    for index in range(N):
        out_col = dout[index].reshape(F, outH*outW)          #[F x H'*W']
        w_out = w_row.T.dot(out_col)                         #[C*FH*FW x H'*W']
        dx_cur = np.zeros((C, H, W))
        neuron = 0
        for i in range(0, H-FH+1):
            for j in range(0, W-FW+1):
                dx_cur[:,i:i+FH,j:j+FW] += w_out[:,neuron].reshape(C,FH,FW)
                x_col[:,neuron] = x[index,:,i:i+FH,j:j+FW].reshape(C*FH*FW)
                neuron += 1
        dx[index] = dx_cur[:,:,:]
        dw += out_col.dot(x_col.T).reshape(F,C,FH,FW)
        db += out_col.sum(axis=1)

    grads=(dx, dw, db)
    return grads


def fcc_forward_naive(x, w, b):

    out = None
   
    dim_size = x[0].shape
   
    X = x.reshape(int(x.shape[0]), np.prod(dim_size))
    out = X.dot(w) + b
   
    cache = ( out, x, w, b)
    return out, cache


def fcc_backward_naive(dout, cache):
   
    y,x, w, b = cache
    dx, dw, db = None, None, None
  
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)

    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)

    grads=(dx, dw, db)

    return grads

def relu_forward(x):
 
    out = None
    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout * (x > 0)

    return dx
