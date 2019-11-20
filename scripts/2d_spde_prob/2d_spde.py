from __future__ import division
import argparse
  
#################################################################
# ======================
## note the code is based on cell centers
# ======================
#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-train_data', dest = 'train_data', type = str, 
                    default = 'data/train_exp_nx1=32_nx2=32_lx1=0.05_lx2=0.08_v=0.75_num_samples=10000.npy', help  = 'Training data file.')
parser.add_argument('-test_data', dest = 'test_data', type = str, 
                    default = 'data/test_exp_nx1=32_nx2=32_lx1=0.05_lx2=0.08_v=0.75_num_samples=2000.npy', help  = 'Testing data file.')
parser.add_argument('-nx1', dest = 'nx1', type = int, 
                    default = 32, help  = 'Number of FV cells in the x1 direction.')# number of cells/cellcenters/pixels/pixelcenters in x1-direction
parser.add_argument('-nx2', dest = 'nx2', type = int, 
                    default = 32, help  = 'Number of FV cells in the x2 direction.')# number of cells/cellcenters/pixels/pixelcenters in x2-direction


parser.add_argument('-DNN_type', dest = 'DNN_type', type = str, 
                    default = 'Resnet', help  = 'Type of DNN (Resnet:Residual network, FC:Fully connected network).')
parser.add_argument('-n', dest = 'n', type = int, 
                    default = 64, help  = 'Number of neurons in each block.')
parser.add_argument('-num_block', dest = 'num_block', type = int, 
                    default = 1, help  = 'Number of blocks.')
parser.add_argument('-d', dest = 'd', type = str, 
                    default = '[5,5]', help  = 'Number of neurons per layer.')
parser.add_argument('-act_func', dest = 'act_func', type = str, 
                    default = 'swish', help  = 'Activation function.')


parser.add_argument('-loss_type', dest = 'loss_type', type = str, 
                    default = 'EF', help  = 'Type of Loss to use for training (EF: Energy Functional, SR: Squared Residual).')
parser.add_argument('-lr', dest = 'lr', type = float, 
                    default = 0.001, help  = 'Learning rate.')
parser.add_argument('-max_it', dest = 'max_it', type = int, 
                    default = 1000, help  = 'Maximum number of iterations.')
parser.add_argument('-M_A', dest = 'M_A', type = int, 
                    default = 10, help  = 'Batch size: number of input field images in each iteration.')
parser.add_argument('-M_x', dest = 'M_x', type = int, 
                    default = 10, help  = 'Batch size: number of x-samples=[x1,x2] on each of the sampled image in each iteration.') ## M_x cannot be greater than nx1*nx2 (FOR THIS CODE)


parser.add_argument('-seed', dest = 'seed', type = int, 
                    default = 0, help  = 'Random seed number.') # seed for reproducability
parser.add_argument('-variation', dest = 'variation', type = str, 
                    default = 'a', help  = 'Model variation currently trying.')

args = parser.parse_args()

#################################################################
import matplotlib
matplotlib.use('PS')
import tensorflow as tf
import random 
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'

seed = args.seed
# Setting the seed for numpy-generated random numbers
np.random.seed(seed=seed)

# Setting the seed for python random numbers
random.seed(seed)

# Setting the graph-level random seed.
tf.set_random_seed(seed)

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model
from keras.layers import Dense, Activation, Input, concatenate, Lambda, Add
from keras.utils import plot_model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import matplotlib.pyplot as plt 
import GPy
from fipy import *
# import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 300
import seaborn as sns 
sns.set_context('talk')
sns.set_style('white')
from pdb import set_trace as keyboard
import sys
import time 
#################################################################
# ------------------------------------------------------------

# loading data
train_data = np.load(os.path.join(os.getcwd(),args.train_data))
test_data = np.load(os.path.join(os.getcwd(),args.test_data))

# bounding input fields from below and above
lower_bound =  np.exp(-5.298317366548036) # 0.005000000000000002
upper_bound =  np.exp(3.5) # 33.11545195869231

train_data = np.where(train_data < lower_bound, lower_bound,train_data) 
train_data  = np.where(train_data > upper_bound, upper_bound, train_data)  

test_data = np.where(test_data < lower_bound, lower_bound,test_data) 
test_data  = np.where(test_data > upper_bound, upper_bound, test_data)  

# ------------------------------------------------------------

nx1 = args.nx1
nx2 = args.nx2

DNN_type = args.DNN_type
n = args.n
num_block = args.num_block
d = args.d
act_func = args.act_func

loss_type = args.loss_type
lr = args.lr
max_it = args.max_it
M_A = args.M_A
M_x = args.M_x

variation = args.variation 

# ------------------------------------------------------------
# needs to be defined as activation class otherwise error
# AttributeError: 'Activation' object has no attribute '__name__'    
class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})
# ------------------------------------------------------------
# BUILDING DNN APPROXIMATOR
# ======================
# DNN network i/p:x1,x2,A o/p:prediction

x1 = Input(shape=(1,))
x2 = Input(shape=(1,))
A = Input(shape=(nx1*nx2,)) # input field image: conductivity image
a_val = Input(shape=(1,)) # input field value: conductivity value at the corresponding 'input(x1,x2)' location

if DNN_type == 'Resnet':
    x1_x2_A = concatenate([x1,x2,A])
    o = Dense(n)(x1_x2_A)
    for i in range(num_block):
        z = Dense(n, activation = act_func)(o)
        z = Dense(n, activation = act_func)(z)
        o = Add()([z, o])
    prediction = Dense(1)(o)
    print DNN_type

elif DNN_type == 'FC':
    num_neurons_per_layer = map(int, d.strip('[]').split(','))
    x1_x2_A = concatenate([x1,x2,A])
    z = Dense(num_neurons_per_layer[0], activation=act_func)(x1_x2_A)
    for n in num_neurons_per_layer[1:]:
        z = Dense(n, activation=act_func)(z)
    prediction = Dense(1)(z)
    print DNN_type

def myFunc(t):
    return ((1-t[0])+(t[0]*(1-t[0])*t[2])) # dirichlet on left and right sides of plate

u = Lambda(myFunc, output_shape=(1,))([x1,x2,prediction]) # field of interest : temperature
model = Model(inputs=[x1,x2,A], outputs=u)

# BUILDING LOSS FUNCTIONS
# ======================

dudx1 = tf.gradients(u, x1)[0] 
dudx2 = tf.gradients(u, x2)[0] 

tf_a = a_val

c = 0.
f = 0. # source

# loss function
# ======================
if loss_type == 'EF':
    term_1 = 0.5 * ((tf_a * (dudx1 ** 2 + dudx2 ** 2)) + (c*u*u))
    V = term_1 - (f * u)
    ef_loss = tf.reduce_sum(V)/(M_x * M_A)
    #or
    # ef_loss = tf.reduce_mean(V)
    loss = ef_loss 
    print('loss energy functional form')
    # create directories
    resultdir = os.path.join(os.getcwd(), 'results','loss_EF_form','DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation)) 

elif loss_type == 'SR':
    residual = -(tf.gradients(tf_a * dudx1, x1)[0]) - (tf.gradients(tf_a * dudx2, x2)[0]) + (c*u) - f
    sqresi_loss = tf.reduce_sum(tf.square(residual))/(M_x * M_A)   
    #or
    # sqresi_loss = tf.reduce_mean(tf.square(residual)) 
    loss = sqresi_loss 
    print('loss squared residual form')
    # create directories
    resultdir = os.path.join(os.getcwd(), 'results','loss_SR_form','DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation)) 

if not os.path.exists(resultdir):
    os.makedirs(resultdir)

# ======================
orig_stdout = sys.stdout
q = open(os.path.join(resultdir, 'loss_output='+str(DNN_type)+'_'+str(nx1*nx2)+'_'+str(seed)+'_'+str(variation)+'.txt'), 'w')
sys.stdout = q
start = time.time()
print ("------START------")
print args.train_data
print args.test_data

if DNN_type == 'Resnet':
    print (nx1,nx2,DNN_type,n,num_block,act_func,loss_type,lr,max_it,M_A,M_x,seed,variation)
elif DNN_type == 'FC':
    print (nx1,nx2,DNN_type,num_neurons_per_layer,act_func,loss_type,lr,max_it,M_A,M_x,seed,variation)

plot_model(model, to_file=os.path.join(resultdir,'stoch_heq_nn_fipy.pdf'))
# ======================

train = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()   
sess = tf.Session()
K.set_session(sess)
sess.run(init) 

I=[]
Loss=[]

weights = sess.run(model.weights)
# print weights
w=[]
[w.extend(weights[j].flatten()) for j in range(len(weights))]
# print len(w)
plt.hist(w, bins=20)
plt.title('Histogram_Weights&biases_all_layers_before_training')
plt.savefig(os.path.join(resultdir,'Histogram_Weights&biases_all_layers_before_training.pdf'))
plt.pause(1)
plt.close()

# ======================
print ("--------------------------------------------------------------")

#defining mesh to get cellcenters
Lx1 = 1.  # always put . after 1 
Lx2 = 1.  # always put . after 1 
mesh = Grid2D(nx=nx1, ny=nx2, dx=Lx1/nx1, dy=Lx2/nx2) # with nx1*nx2 number of cells/cellcenters/pixels/pixelcenters
cellcenters = mesh.cellCenters.value.T # (nx1*nx2,2) matrix
# print cellcenters

gridnum_list = [i for i in range(nx1*nx2)] # grid at cell centers
# print gridnum_list
# print np.shape(gridnum_list)
# print type(gridnum_list)
print ('*****')

print ("--------------------------------------------------------------")

for i in range(max_it):
    # Get a batch of points
    X1i_final=np.zeros((1, 1)) # sampled x's X1 coordinates
    X2i_final=np.zeros((1, 1)) # sampled x's X2 coordinates
    AAi_final = np.zeros((1, cellcenters.shape[0])) # images of input field: conductivity # or np.zeros((1, nx1*nx2)) 
    Ai_val_final = np.zeros((1, 1)) # input field values at sampled x's
    
    for t in xrange(M_A):

        # sampling grid locations from gridnum_list
        gridnum_sam = np.random.choice(gridnum_list, size=M_x, replace=False, p=None) # p: The probabilities associated with each entry in a.\
                                                               # If not given the sample assumes a uniform distribution over all entries in a.   
                                                               # replace: Whether the sample is with or without replacement
        # print gridnum_sam
        # print type(gridnum_sam)
        # print np.shape(gridnum_sam)

        # sampled x's coordinates
        Xi = np.ndarray((M_x, 2)).astype(np.float32)
        for j in range(M_x):
            Xi[j, :] = cellcenters[gridnum_sam[j],:]
        # print Xi
        # print ('########')
        X1i_final = np.vstack((X1i_final,Xi[:,0][:,None]))
        X2i_final = np.vstack((X2i_final,Xi[:,1][:,None]))

        # getting input field images
        Ai = train_data[ random.randint(0, np.shape(train_data)[0]-1), : ].reshape(1,-1)
        # Ai is one image of input field: conductivity of nx1*nx2 cells/cellcenters/pixels/pixelcenters picked from train_data # returns (1 , nx1*nx2) matrix 
        # print Ai
        # print ('########')
        AAi = np.repeat(Ai, np.shape(Xi)[0], axis=0) # just repeating 
        AAi_final = np.vstack((AAi_final,AAi))
        
        # getting input field values at sampled x's to use them in loss function calculations
        Ai_val = np.ndarray((M_x, 1)).astype(np.float32) # or np.ndarray((np.shape(Xi)[0], 1)).astype(np.float32)
        for g in range(M_x): # or for g in range(np.shape(Xi)[0]):
            Ai_val[g,0] = Ai[0,gridnum_sam[g]] 
        # print Ai_val
        # print ('#################################')
        Ai_val_final = np.vstack((Ai_val_final,Ai_val))

    X1i_final=np.delete(X1i_final, (0), axis=0) #to delete the first row
    X2i_final=np.delete(X2i_final, (0), axis=0) #to delete the first row
    AAi_final = np.delete(AAi_final, (0), axis=0) #to delete the first row
    Ai_val_final = np.delete(Ai_val_final, (0), axis=0) #to delete the first row
    
    # print X1i_final 
    # print X2i_final
    # print AAi_final
    # print Ai_val_final
    # print ('done')

    sess.run(train, feed_dict={x1:X1i_final, x2:X2i_final, A:AAi_final, a_val:Ai_val_final})
    l = sess.run(loss, feed_dict={x1:X1i_final, x2:X2i_final, A:AAi_final, a_val:Ai_val_final})
    
    I.append(i)
    Loss.append(l)

    # display
    if i % 500 == 0:
        print ("Iteration: "+str(i)+"; Train loss:"+str(l)+";")
        # weights = sess.run(model.weights)
        # w=[]
        # [w.extend(weights[j].flatten()) for j in range(len(weights))]
        # # print len(w)
        # plt.hist(w, bins=20)
        # plt.title('Iteration:='+str(i)+'_ Histogram_Weights&biases_all_layers')
        # plt.savefig(os.path.join(resultdir,'Iteration='+str(i)+'_ Histogram_Weights&biases_all_layers.pdf'))
        # plt.pause(1)
        # plt.close()
        # # keyboard()
print ("--------------------------------------------------------------")

weights = sess.run(model.weights)
# print weights
w=[]
[w.extend(weights[j].flatten()) for j in range(len(weights))]
# print len(w)
plt.hist(w, bins=20)
plt.title('Histogram_Weights&biases_all_layers_after_training')
plt.savefig(os.path.join(resultdir,'Histogram_Weights&biases_all_layers_after_training.pdf'))
plt.pause(1)
plt.close()
model.summary()
model.save(os.path.join(resultdir,'my_model.h5'))
model.save_weights(os.path.join(resultdir,'my_model_weights.h5'))

plt.plot(I, Loss, 'blue', lw=1.5, label='Iteration_vs_Trainloss')
plt.xlabel('Iteration')
plt.ylabel('Trainloss')
plt.title('DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation)+'_ Iteration Vs Trainloss')
plt.savefig(os.path.join(resultdir,'DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation)+'_ Iteration Vs Trainloss.pdf'))
plt.tight_layout()
plt.legend(loc='best');
plt.pause(5)
plt.close()

np.save(os.path.join(resultdir,'I.npy'), np.asarray(I))
np.save(os.path.join(resultdir,'Loss.npy'), np.asarray(Loss))
# np.load(os.path.join(resultdir,'I.npy'))
# np.load(os.path.join(resultdir,'Loss.npy'))

# end timer
finish = time.time() - start  # time for network to train

# TESTING(checking NN solution against fipy solution) 
# ======================
# test cases
nsamples = np.shape(test_data)[0]
# validation error and relative RMS error
val = []
rel_RMS_num = []
rel_RMS_den = []
# get all relative errrors and r2 scores
relerrors = []
r2scores = []
# get all things for plots
samples_inputfield = np.zeros((nsamples, nx1*nx2))
samples_u_DNN = np.zeros((nsamples, nx1*nx2))
samples_u_fipy = np.zeros((nsamples, nx1*nx2))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["navy","blue","deepskyblue","limegreen","yellow","darkorange","red","maroon"])

np.random.seed(23)
for i in range(nsamples): # test cases
###############################################################
    # FIPY solution
    value_left=1.
    value_right=0.
    value_top=0.
    value_bottom=0.

    Lx1 = 1.  # always put . after 1 
    Lx2 = 1.  # always put . after 1 

    # define mesh
    mesh = Grid2D(nx=nx1, ny=nx2, dx=Lx1/nx1, dy=Lx2/nx2) # with nx1*nx2 number of cells/cellcenters/pixels/pixelcenters
 
    # define cell and face variables
    phi = CellVariable(name='$T(x)$', mesh=mesh, value=0.)
    D = CellVariable(name='$D(x)$', mesh=mesh, value=1.0) ## coefficient in diffusion equation
    # D = FaceVariable(name='$D(x)$', mesh=mesh, value=1.0) ## coefficient in diffusion equation
    source = CellVariable(name='$f(x)$', mesh=mesh, value=1.0)
    C = CellVariable(name='$C(x)$', mesh=mesh, value=1.0)

    # apply boundary conditions
    # dirichet
    phi.constrain(value_left, mesh.facesLeft)
    phi.constrain(value_right, mesh.facesRight)
    
    # homogeneous Neumann
    phi.faceGrad.constrain(value_top, mesh.facesTop)
    phi.faceGrad.constrain(value_bottom, mesh.facesBottom)
    
    # setup the diffusion problem
    eq = -DiffusionTerm(coeff=D)+ImplicitSourceTerm(coeff=C) == source

    source.setValue(f)
    C.setValue(c)

    # getting input field images
    a = test_data[ i , : ].reshape(-1,1)
    # 'a' is one image of input field: conductivity image of nx1*nx2 cells/cellcenters/pixels/pixelcenters from test_data # returns (nx1*nx2,1) matrix 
    D.setValue(a.ravel())

    eq.solve(var=phi)
    x_fipy = mesh.cellCenters.value.T ## fipy solution (nx1*nx2,2) matrix # same as cellcenters defined above
    u_fipy = phi.value[:][:, None] ## fipy solution  (nx1*nx2,1) matrix

    x1_f = x_fipy[:,0][:,None] # x1-coordinates of cell centers (nx1*nx2,1) matrix
    x2_f = x_fipy[:,1][:,None] # x2-coordinates of cell centers (nx1*nx2,1) matrix
    # print ('done1')
###############################################################
    #Neuralnet solution
    u_DNN = sess.run(u, feed_dict={x1:x1_f, x2:x2_f,  A:np.repeat(a.T, np.shape(x1_f)[0], axis=0)})
    # print ('done2')
#################################################################
    val.append(np.sum((u_fipy-u_DNN)**2, axis=0)) 
    rel_RMS_num.append(np.sum((u_fipy-u_DNN)**2, axis=0))
    rel_RMS_den.append(np.sum((u_fipy)**2, axis=0))
###############################################################
    from sklearn import metrics
    r2score = metrics.r2_score(u_fipy.flatten(), u_DNN.flatten()) 
    relerror = np.linalg.norm(u_fipy.flatten() - u_DNN.flatten()) / np.linalg.norm(u_fipy.flatten())
    r2score = float('%.4f'%r2score)
    relerror = float('%.4f'%relerror)
    relerrors.append(relerror)
    r2scores.append(r2score)   
###############################################################
    samples_inputfield[i] = a.ravel()
    samples_u_DNN[i] = u_DNN.flatten()
    samples_u_fipy[i] = u_fipy.flatten()   
###############################################################
    if i<=20:
        # Initialize the plot
        fig = plt.figure(figsize=(18,6))

        try:
            ax1.lines.remove(c1[0])
            ax2.lines.remove(c2[0])
            ax3.lines.remove(c3[0])

        except:
            pass

        ax1 = fig.add_subplot(1, 3, 1)
        c1 = ax1.contourf( x1_f.reshape((nx1,nx2)), x2_f.reshape((nx1,nx2)), np.log(a).reshape((nx1,nx2)), 100, cmap=cmap) # set levels automatically
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c1.collections:
            j.set_edgecolor("face")
        plt.colorbar(c1)
        plt.title('$log(Input \ field)$', fontsize=14)
        plt.xlabel('$x1$', fontsize=12)
        plt.ylabel('$x2$', fontsize=12)


        ax2 = fig.add_subplot(1, 3, 2)
        c2 = ax2.contourf( x1_f.reshape((nx1,nx2)), x2_f.reshape((nx1,nx2)), u_fipy.reshape((nx1,nx2)), 100, cmap=cmap) # set levels as previous levels
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c2.collections:
            j.set_edgecolor("face")
        plt.colorbar(c2)
        plt.title('$FVM \ solution$',fontsize=14)
        plt.xlabel('$x1$', fontsize=12)
        plt.ylabel('$x2$', fontsize=12)


        ax3 = fig.add_subplot(1, 3, 3)
        c3 = ax3.contourf( x1_f.reshape((nx1,nx2)), x2_f.reshape((nx1,nx2)), u_DNN.reshape((nx1,nx2)), 100, cmap=cmap) # set levels automatically
        # This is the fix for the white lines between contour levels (https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills)
        for j in c3.collections:
            j.set_edgecolor("face")
        plt.colorbar(c3)
        plt.title('$DNN \ solution: \ Rel. L_2 \ Error$ = '+str(relerror)+', $R^{2}$ = '+str(r2score), fontsize=14)
        plt.xlabel('$x1$', fontsize=12)
        plt.ylabel('$x2$', fontsize=12)

        plt.tight_layout()
        # plt.suptitle('test_case='+str(i+1)+'_DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation), fontsize=12)
        plt.savefig(os.path.join(resultdir,'test_case='+str(i+1)+'_DNN_type='+str(DNN_type)+'_nx1*nx2='+str(nx1*nx2)+'_seed='+str(seed)+'_'+str(variation)+'_nnpred-fipy.pdf'))     
        plt.show()
        plt.pause(0.1)
print i
print ("--------------------------------------------------------------")
####################################################################################################################
plt.close('all')
# https://stats.stackexchange.com/questions/189783/calculating-neural-network-error
# print val
vali_error = np.sum(val)/(np.shape(val)[0]*np.shape(x_fipy)[0])
print ('validation_error='+str(vali_error))

# https://www.rocq.inria.fr/modulef/Doc/GB/Guide6-10/node21.html
rel_RMS_error = np.sqrt(np.sum(rel_RMS_num)/np.sum(rel_RMS_den))
print ('relative_RMS_error='+str(rel_RMS_error))


np.save(os.path.join(resultdir,'cellcenters.npy'), cellcenters) # or x_fipy

np.save(os.path.join(resultdir,'samples_inputfield.npy'), samples_inputfield)
np.save(os.path.join(resultdir,'samples_u_DNN.npy'), samples_u_DNN)
np.save(os.path.join(resultdir,'samples_u_fipy.npy'), samples_u_fipy)

relerrors = np.array(relerrors)
r2scores = np.array(r2scores)

np.save(os.path.join(resultdir,'relerrors.npy'), relerrors)
np.save(os.path.join(resultdir,'r2scores.npy'), r2scores)

#plt.figure(figsize=(8, 6))
plt.hist(relerrors, alpha = 0.7, bins = 100, normed=True, label='Histogram of Rel. $L_2$ Error')
plt.tight_layout()
plt.legend(loc = 'best', fontsize = 14)
plt.savefig(os.path.join(resultdir,'rel_errors_hist.pdf')) 
plt.close()

plt.hist(r2scores, alpha = 0.7, bins = 100, normed=True, label='Histogram of $R^2$')
plt.tight_layout()
plt.legend(loc = 'best', fontsize = 14)
plt.savefig(os.path.join(resultdir,'r2scores_hist.pdf')) 
plt.close()

print "Time (sec) to complete: " +str(finish) # time for network to train
print ("------END------")
sys.stdout = orig_stdout
q.close()
###############################################################
