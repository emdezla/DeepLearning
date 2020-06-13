# test.py>

import torch
import neural_framework as nf
from tools import *

######################################################################

# DATA SET GENERATION AND PLOT
N = 1000
training_set = generate_dataset(N)       
test_set = generate_dataset(N)
plot_circle_with_labels(training_set,"Training set with correct labels")

######################################################################

# DATA FORMATTING FOR THE NN INPUT
x = torch.from_numpy(training_set[:,0:2])             # Point coordinates
y = torch.from_numpy(training_set[:,2:])              # Labels
x_test = torch.from_numpy(test_set[:,0:2])
y_test = torch.from_numpy(test_set[:,2:])

# NEURAL NETWORK MODEL DEFINITION
D_in, H, D_out = 2,25,2                                # Number of nodes
model = nf.Sequential(                                 # NN architecture
            nf.Linear(D_in,H),nf.ReLU(),               # 1st hidden layer
            nf.Linear(H,H),nf.Tanh(),                  # 2nd hidden layer
            nf.Linear(H,H),nf.ReLU(),                  # 3rd hidden layer
            nf.Linear(H,D_out),                        # output layer
            nf.LossMSE())

# HYPERPARAMETERS 
B = 20                                                 # Batch size
Xb,Yb = x.reshape(-1,B,2),y.reshape(-1,B,2)            # data splitted in batches
learning_rate = 1e-4                                   # Learning rate
n_epochs = 150                                         # Number of epochs
L,Lb,T,E = [],[],[],[]                                 # Lists for logging loss and error

# TRAINING
for t in range(n_epochs):
    for batch,label in zip(Xb,Yb):
        pred = model.forward(batch)                     # Feed-forward
        Lb.append(model.loss(pred,label))               # Train Loss computation
        model.backward()                                # Back-propagation
        model.learn(learning_rate)                      # Weights update    
        
    L.append(sum(Lb)/len(Lb))                           # Average train loss
    T.append(model.error(model.forward(x),y))           # Train error
    E.append(model.error(model.forward(x_test),y_test)) # Test error
    
parameters = model.param(show=1)                        # Show parameters
log_file(L,T,E)                                         # Save evolution
plot_evolution(L,Lb,T,E)                                # Plot evolution

######################################################################

# CHECK PREDICTED LABELS FOR TRAINING SET
training_prediction = model.forward(x)
training_error = model.error(training_prediction,y)
check_result(x,training_prediction,training_error,N,"training")

# CHECK PREDICTED LABELS FOR TEST SET
test_prediction = model.forward(x_test)
test_error = model.error(test_prediction,y_test)
check_result(x_test,test_prediction,test_error,N,"test")

