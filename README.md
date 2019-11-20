# Variational-elliptic-SPDE
## **Simulator-free Solution of High-Dimensional Stochastic Elliptic Partial Differential Equations using Deep Neural Networks**
[SharmilaKarumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en), [RohitTripathy](https://scholar.google.com/citations?user=h4Qyq9gAAAAJ&hl=en), [IliasBilionis](https://scholar.google.com/citations?user=rjXLtJMAAAAJ&hl=en), [JiteshPanchal](https://scholar.google.com/citations?user=fznjeN0AAAAJ&hl=en)

Our paper proposes a novel deep neural network (DNN) based solver for elliptic stochastic partial differential equations (SPDEs) characterized by high-dimensional input uncertainty. Our method is intended to overcome the curse of dimensionality associated with traditional approaches for high-dimensional SPDEs. The solution of the elliptic SPDE is parameterized using a deep residual network (ResNet) and trained on a physics-informed loss function.

#### The novel features of our approach are:

1.	Our method is simulator-free i.e. liberated from the requirement of a deterministic forward solver unlike the existing approaches. 
2.	The DNN is trained by minimizing a physics-informed loss function obtained from deriving a variational principle for the elliptic SPDE. 

The significance of the proposed approach is that it overcomes a fundamental limitation of existing state-of-the-art methodologies – it eliminates the requirement of a numerical solver for the deterministic forward problem whose individual evaluations are potentially very expensive. The proposed methodology is easy to implement and scalable to cases where the uncertain input data is characterized by large stochastic dimensionality. We demonstrate our solver-free approach through various examples where the elliptic SPDE is subjected to different types of high-dimensional input uncertainties. Also, we solve high-dimensional uncertainty propagation and inverse problems.

### Deep Resnet architecture:
<img width="244" alt="Resnet_schematic" src="https://user-images.githubusercontent.com/30219043/69242800-a8d49180-0b6f-11ea-840f-75450e8027aa.png">

### Results:
These figures shows the comparison of the predicted SPDE solution from our trained deep ResNets and a standard finite volume method (FVM) solver for randomly sampled realizations of the input field.  

2D Gaussian Random Field (GRF) of length-scales 0.05, 0.08
![test_case=6_journal-page-001](https://user-images.githubusercontent.com/30219043/69241808-90fc0e00-0b6d-11ea-8b87-0a36d3ca2be9.jpg)
2D Warped GRF
![test_case=25_journal-page-001](https://user-images.githubusercontent.com/30219043/69241842-a07b5700-0b6d-11ea-8422-9c4fea7aba61.jpg)
2D Channelized Field
![test_case=29_journal-page-001](https://user-images.githubusercontent.com/30219043/69241873-ae30dc80-0b6d-11ea-8be3-e7ea83acf6bd.jpg)

More experiments and results can be found in our paper at - 
### **Link for the paper:**
https://doi.org/10.1016/j.jcp.2019.109120

Install dependencies at [requirements.txt](https://github.com/PredictiveScienceLab/variational-elliptic-SPDE/blob/master/requirements.txt) and clone our repository
```
git clone https://github.com/PredictiveScienceLab/variational-elliptic-SPDE.git
cd variational-elliptic-SPDE
```
### Data:
All the input datasets used in the paper such as one dimensional and two dimensional gaussian random fields (GRF) - 2D GRF, Warped GRF, Channelized field and Multiple lengthscales GRF etc. to train the deep resnet approximator can be downloaded from link at [/data/link_to_data.txt](https://github.com/PredictiveScienceLab/variational-elliptic-SPDE/blob/master/data/link_to_data.txt).

### Scripts:
Keras-tensorflow implementation codes for building a DNN approximator of SPDE solution are at [./scripts](https://github.com/PredictiveScienceLab/variational-elliptic-SPDE/tree/master/scripts).

For a 2D SPDE problem do
```
python 2d_spde.py -train_data='train_channel_nx1=32_nx2=32_num_samples=4096.npy' -test_data='test_channel_nx1=32_nx2=32_num_samples=512.npy' -nx1=32 -nx2=32 -DNN_type='Resnet' -n=300 -num_block=3 -act_func='swish' -loss_type='EF' -lr=0.0001 -max_it=75000 -M_A=100 -M_x=20 -seed=0
```
* ```-train_data``` = Training data file
* ```-test_data```  = Testing data file
* ```-nx1```  = Number of grid discretizations in the x1 direction.
* ```-nx2```  = Number of grid discretizations in the x2 direction.
* ```-DNN_type```  = Type of DNN (Resnet:Residual network, FC:Fully connected network)
* ```-num_block```  = Number of blocks in the Resnet
* ```-n```  = Number of neurons in each block of a Resnet
* ```-act_func```  = Activation function used
* ```-loss_type```  = Type of Loss to use for training (EF: Energy Functional, SR: Squared Residual)
* ```-lr```  = Learning rate used
* ```-max_it```  = Maximum number of iterations to train the DNN
* ```-M_A```  =  Number of input field images to be picked in each iteration
* ```-M_x```  =  Number of x=[x1,x2] locations to be picked on each of the picked image
* ```-seed```  =  seed number for reproducability

Results are saved at ```./results``` 

Similar procedure as above for building a DNN approximator for the 1D SPDE problem.

### Trained models:
Download the trained model weights of 1D and 2D SPDE problem from [./trained_models](https://github.com/PredictiveScienceLab/variational-elliptic-SPDE/tree/master/trained_models). Corresponding Resnet architecture and input dataset files used to train the Resnet are included in the 'data_file_and_dnn_architecture.txt' .

### Acknowledgements:
We would like to acknowledge support from the Defense Advanced Research Projects Agency (DARPA) under the Physics of Artificial Intelligence (PAI) program (contract HR00111890034).

### Citation:
If you use this code for your research, please cite our paper https://doi.org/10.1016/j.jcp.2019.109120.
