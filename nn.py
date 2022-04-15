import numpy as np
from losses import *
from layers import *

class Neural_Net():

    def __init__(self):

        self.layers=[]
        self.nlayers=0
        self.layer_weights=[]
        self.layer_biases=[]
        self.layer_cache=[]
        self.layer_grads=[]


    def add_fcc(self,input_shape,output_shape):
        
        self.layers.append("fcc")
        self.layer_weights.append(np.random.rand(input_shape,output_shape))
        self.layer_biases.append(np.random.rand(output_shape))
        self.layer_cache.append(None)
        self.layer_grads.append(None)
        self.nlayers+=1
    
    def add_conv(self,F,C,H,W):
        
        self.layers.append("conv")
        self.layer_weights.append(np.random.rand(F,C,H,W))
        self.layer_biases.append(np.random.rand(F))
        self.layer_cache.append(None)
        self.layer_grads.append(None)
        self.nlayers+=1

    def add_relu(self):
        self.layers.append("relu")
        self.layer_weights.append(None)
        self.layer_biases.append(None)
        self.layer_cache.append(None)
        self.layer_grads.append(None)
        self.nlayers+=1
    

    def fwprop(self,x):

        iter=0
   

        for layer in self.layers:
            if layer=="fcc":
                x, cache=fcc_forward_naive(x,self.layer_weights[iter],self.layer_biases[iter])
            elif layer=="conv":
                x,cache=conv_forward_naive(x,self.layer_weights[iter],self.layer_biases[iter])
            elif layer=="relu":
                x,cache=relu_forward(x)

            self.layer_cache[iter]=cache
            iter+=1

        return x

    def bckprop(self,dout):

        iter=self.nlayers-1

        for layer in reversed(self.layers):
            if layer == "fcc":
                grads=fcc_backward_naive(dout,self.layer_cache[iter])
                self.layer_grads[iter]=grads
                dout,_,_ = grads
            elif layer == "conv":
                grads=conv_backward_naive(dout,self.layer_cache[iter])
                self.layer_grads[iter]=grads
                dout,_,_ = grads
            elif layer == "relu":
                grads = relu_backward(dout,self.layer_cache[iter])
                dout=grads
           

            self.layer_grads[iter]=grads
            iter-=1

    def update_weights(self,lr):
        for i in range(self.nlayers):
            # print(self.layer_grads[i])
            if self.layer_weights[i] is not None:
                self.layer_weights[i]-=lr*self.layer_grads[i][1]
                self.layer_biases[i]-=lr*self.layer_grads[i][2]

    def loss(self,y,t):
        loss, dx= softmax_loss(y,t)

        return loss, dx
       
    def train(self,X,Y,lr,epochs):
        for epoch in range(epochs):
            # print("Epoch: ",epoch)
            self.fwprop(X)
            loss,dout= softmax_loss(self.layer_cache[-1],Y)
            self.bckprop(dout)
            self.update_weights(lr)
            print("Epoch: ",epoch,"/",epochs,"\tLoss: ",loss)
            print("\n")



