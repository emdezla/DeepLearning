# tools.py>

import matplotlib. pyplot as plt
plt.style.use('seaborn')
import numpy as np
import csv

######################################################################

def generate_dataset (N):
    z = np.zeros((N,4))
    z[:,0] = np.float32(np.random.uniform(0,1, N)) # x-coordinate for points
    z[:,1] = np.float32(np.random.uniform(0,1, N)) # y-coordinate for points
    x_C, y_C, r_C = 0.5, 0.5, 1/np.sqrt(2*np.pi)   # circle definition
    z[:,2] = np.float32(np.sqrt((z[:,0]-x_C)**2+(z[:,1]-y_C)**2)<r_C) # 1 if inside circle
    z[:,3] = np.float32(np.sqrt((z[:,0]-x_C)**2+(z[:,1]-y_C)**2)>r_C) #1 if outside circle
    
    return z

######################################################################

def set_plot_param(ax,xlabel,ylabel,title,size=20):
    ax.set_xlabel(xlabel, fontsize = size)
    ax.set_ylabel(ylabel, fontsize = size)  
    ax.tick_params(axis="x", labelsize=size)
    ax.tick_params(axis="y", labelsize=size)
    ax.set_title(title, fontsize=size)

######################################################################

def plot_circle_with_labels (z,title):
    x_cor, y_cor,r_in = z[:,0],z[:,1],z[:,2]    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_cor[r_in==1],y_cor[r_in==1],'.' ,color='blue')
    ax.plot(x_cor[r_in==0],y_cor[r_in==0],'.' ,color='red')
    circle = plt.Circle((0.5, 0.5), 1/np.sqrt(2*np.pi), 
                        color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    set_plot_param(ax,'X axis','Y axis',title)
    fig.savefig(title+'.png')
    
 ######################################################################

def plot_evolution(loss, real_loss, train_error, test_error):
    
    #Average train loss evolution
    fig, ax = plt.subplots(figsize=(8,6))
    n_epochs = len(loss)
    ax.plot( list(range( n_epochs ) ),loss)
    set_plot_param(ax,"Epoch",'MSE Loss','Average loss evolution')
    fig.savefig('train_loss_average.png')
    
    #Real train loss evolution over selected epoch
    fig, ax = plt.subplots(figsize=(8,6))
    iterations = list(range(int(len(real_loss)/n_epochs)))
    sampled_epochs =list(range(n_epochs))[::30]
    for i in sampled_epochs:
        ax.plot(iterations,real_loss[len(iterations)*i:len(iterations)*(i+1)],label="Epoch {}".format(i))
    set_plot_param(ax,"Iteration",'MSE Loss','Train Loss evolution over an epoch')
    ax.legend(fontsize=20)
    fig.savefig('train_loss_batch.png')
    
    #Train error evolution compared with test error
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(list(range(n_epochs)),train_error, label="Train error",color='magenta',linewidth='2.5')
    ax.plot(list(range(n_epochs)),test_error, label="Test error",color='limegreen',linewidth='2.5')
    set_plot_param(ax,"Epoch",'Error','Error evolution')
    ax.legend(fontsize=20)
    fig.savefig('train_error.png')

    
######################################################################

def check_result (x,y,error,N,title):
    result = np.zeros((N,3))
    x.reshape(N,-1)
    y.reshape(N,-1)
    result[:,0:2] = x
    result[:,2] =np.round(y[:,0].numpy())
    string_error = "(error = {:.2f} %)".format(error*100)
    info = "Predicted labels for "+title+" set "
    plot_circle_with_labels(result,info+string_error)

######################################################################

def log_file (L,T,E,filename="log_file.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train_Loss", "Train_Error","Test_Error"])
        for e in range(len(L)):
            writer.writerow([e, round(L[e].item(),3),round(T[e],3),round(E[e],3)])
