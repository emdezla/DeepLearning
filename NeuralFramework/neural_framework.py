# neural_framework.py>

import torch, time
torch.set_grad_enabled ( False )

######################################################################

class Module ():
    def forward (self, x ):
        raise NotImplementedError("Must be implemented by subclass")
    def backward (self , grad_output ):
        raise NotImplementedError ("Must be implemented by subclass")
    def param ( self ):
        return []
    
######################################################################

class ReLU (Module):
    def __init__ (self):
        self.type = "ReLU activation"
    def forward (self , x):
        self.x = x
        return x.clamp(min=0)
    def backward (self , grad_output):
        grad_input = grad_output.clone()
        grad_input[self.x<0] = 0
        return grad_input
    
######################################################################

class Tanh (Module):
    def __init__ (self):
        self.type = "Tanh activation"
    def forward (self , x):
        self.result = 2/(1+torch.exp(-2*x))-1
        return  self.result
    def backward (self , grad_output):
        grad_activation = 1 - self.result**2
        return grad_activation*grad_output
    
######################################################################

class Sigmoid (Module):
    def __init__ (self):
        self.type = "Sigmoid activation"
    def forward (self , x):
        self.result =  1/(1+torch.exp(-x))
        return  self.result
    def backward (self , grad_output):
        grad_activation = self.result*(1 - self.result)
        return grad_activation*grad_output
    
######################################################################

class Linear (Module):
    def __init__ (self,Din,Dout):
        self.Din = Din
        self.type = "Linear layer with {} nodes".format(Dout)
        self.w = torch.randn(Din,Dout).type(torch.DoubleTensor)*torch.sqrt(torch.tensor(2/Din))
        self.b = torch.zeros(1,Dout).type(torch.DoubleTensor)        
    def forward (self , x):
        self.x = x
        self.uno = torch.ones(len(self.x),1).type(torch.DoubleTensor)
        return x.mm(self.w) + self.uno.mm(self.b)
    def backward (self , grad_output):
        self.gradw = self.x.t().mm(grad_output)
        self.gradb = self.uno.t().mm(grad_output)
        grad_input = grad_output.mm(self.w.t())
        return grad_input
    def param ( self ):
        return [[self.w,self.gradw],[self.b,self.gradb]]
    def optimise (self,learning_rate):
        self.w -= learning_rate*self.gradw
        self.b -= learning_rate*self.gradb

######################################################################

class LossMSE (Module):
    def __init__ (self):
        self.type = "Mean Squared Error"
    def forward (self , y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return torch.mean((y_pred - self.y_true)**2)
    def backward (self):
        return 2.0 *(self.y_pred - self.y_true)

######################################################################

class Sequential (Module):
    def __init__(self, *args):
        self.modules = []
        print("NEURAL NETWORK ARCHITECTURE:")
        for idx, module in enumerate(args):
            if (idx == 0): 
                print("** Input layer with {} nodes".format(module.Din))
            if idx + 1 == len(args):
                self.last = module
            else:
                self.modules.append(module)
            print("**",module.type)
    def forward (self,x):
        #print("Feed-forward: ", end =" ")
        self.x = x
        for mod in self.modules:
            self.x = mod.forward(self.x)
        #print("Done")
        return self.x 
    def backward (self):
        #print("Back-propagation: ", end =" ")
        self.grad_io = self.last.backward()
        for mod in reversed(self.modules):
            self.grad_io = mod.backward(self.grad_io)
        #print("Done")
        return self.grad_io
    def param (self,show=0):
        parameters = []
        if show:
            print("NEURAL NETWORK PARAMETERS:")
        for idx,mod in enumerate(self.modules):
            param = mod.param()
            parameters.append(param)
            if show:
                if param:
                    total = len(param[0][0])*len(param[0][0][0])+ len(param[1][0])*len(param[1][0][0])
                    print("**Layer",int(1+idx/2),'has', total ,'parameters')
                    print("-----Weights: ", len(param[0][0]),'x',len(param[0][0][0])) 
                    print("-----Biases : ", len(param[1][0]),'x',len(param[1][0][0]))     
        return parameters
    def learn(self,rate):
        #print("Weight-optimisation: ", end =" ")
        for mod in self.modules:
            if mod.param():
                mod.optimise(rate)
        #print("Done")
    def loss(self,prediction,label):
        #print("MSE-Loss computation: ", end =" ")
        output_loss = self.last.forward(prediction,label)
        #print("Done")
        return output_loss
    def error (self,pred,label):
        equal = torch.eq(label,torch.round(pred))
        nb_errors = torch.sum(equal==False) 
        error= float(nb_errors.item())/(2*len(label))
        return error
    


######################################################################

