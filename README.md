# variational-elliptic-SPDE
## **Simulator-free Solution of High-Dimensional Stochastic Elliptic Partial Differential Equations using Deep Neural Networks**


Our paper proposes a novel deep neural network (DNN) based solver for elliptic stochastic partial differential equations (SPDEs) characterized by high-dimensional input uncertainty. Our method is intended to overcome the curse of dimensionality associated with traditional approaches to response surface modeling for high-dimensional SPDEs. The solution of the elliptic SPDE is parameterized using a deep residual network (ResNet) and trained on a physics-informed loss function.

#### The novel features of our approach are:

1.	Our method is simulator-free i.e.  liberated from the requirement of a deterministic forward solver unlike the existing approaches. 
2.	The DNN is trained by minimizing a physics-informed loss function obtained from deriving a variational principle for the elliptic SPDE. 

The significance of the proposed approach is that it overcomes a fundamental limitation of existing state-of-the-art methodologies for non-intrusive response surface modeling – it eliminates the requirement of a numerical solver for the deterministic forward problem whose individual evaluations are potentially very expensive. The proposed methodology is easy to implement and scalable to cases where the uncertain input data is characterized by large stochastic dimensionality. We demonstrate our approach on the steady state heat equation, in 1 and 2 dimensions, with random conductivity characterized by stochastic dimensionality of the order of 100 and 200 respectively.

### **Link for the paper:**
https://arxiv.org/abs/1902.05200

Codes for the three examples in the paper will be made available after the paper is published.