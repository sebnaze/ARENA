# _ARENA_

# Chapter 0: Fundamentals

### Neural Networks
- What makes neural networks more powerful than basic statistical methods like linear regression?

&emsp;It can learn arbitrarily complex associations, less subject to geometrical constraint (e.g. linear).

- What are the advantages of ReLU activations over sigmoids?

&emsp;Faster computation (point-wise linear), easier gradient computation (—> avoids vanishing gradients because un-saturated).

### Linear Algebra
- What is the problem in trying to create a neural network using only linear transformations?

&emsp;It won't predict complex non-linear relationships. 

- Matrices A and B have shapes (n,m) and (m,l). What is the maximum possible rank of the matrix AB?

&emsp;$\text{min}(n,m,l)$

### Probablity & stats

- What is the expected value and variance of the sum of two independent normally distributed random variables?

$$X_1  \sim \mu_1, \sigma_1 \quad \quad X_2 \sim \mu_2, \sigma_2$$

- What is the derivative of quadratic loss $L(x,y)=\frac{1}{2}(x−y)^2$ wrt the input $x$ (assuming all variables are scalars, not vectors)? How about for cross entropy loss $L(x,y)=−(y \, \text{log} \, x+(1−y) \, \text{log}(1−x))$, assuming that $x$∈(0,1) and y is a binary classification label with value either zero or one? What will be the qualitative behaviour of performing gradient descent on x with these loss functions?

&emsp; **MSE loss**: The derivative of $L$ w.r.t $x$ is $x-y$. Qualitatively it means that the gradient is linearly proportional to the error.


&emsp; **Binary Cross Entropy loss**: The derivative of $L$ w.r.t to $x$ is $-\frac{y}{x}+\frac{1-y}{1-x}$, which is $-\frac{1}{x}$ when $y=1$ and $\frac{1}{1-x}$ when $y=0$. Qualitatively, it means the gradient increases/decreases by the inverse of the error in the desired direction, and can be very large when $x$ is close to $1-y$ (i.e. non-linear).

### Information Theory

$$D_{\text{KL}}(P||Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)} = p \log p - p \log q$$

i.e. "difference in entropy" between $P$ and $Q$. 

It can be re-written in terms of cross entropy $H(p,q) = H(p) + D_{\text{KL}}(p||q)$ where $H(p)$ is the entropy of $P \quad  \longrightarrow \quad D_{\text{KL}}(p||q) = H(p,q) - H(p)$. 

Standard cross-entropy equation is $H(p,q) = - \sum p \log q$.

- Suppose $P$ is the probability distribution of which word comes next in natural language, and $Q$ is a language model's estimated probability distribution. What will the cross entropy $H(P,Q)$ be if the model is guessing words uniformly? What will the cross entropy be if the model can predict words with the exact right frequency?

&emsp; If model is guessing uniformly, then $H(p,q) = - \sum p \log q = - \sum p \log \frac{1}{|V|} = \log |V| \sum p = \log |V|$ where V is the vocab set, i.e. $|V|$ is the size of the vocab.  

&emsp; If the model is guessing similarly as $P$, $H(p,q) = -\sum p \log q = - \sum p \log p = H(P)$.


### Programming

#### PyTorch

- what is `nn.Parameter` and `nn.Module`?

A parameter is sub-class of Tensor that contains learnable values that gets updated durng training. You assign a Parameter in a Module. 
Module is a base class for all neural network object, it keeps track of parameters, implements the forward method, etc. 

- when do you call `.backward()`? where are the gradients stored?

You call backward at the end of every batch in training. The Gradients are stored in the `.grad` attribute of each leaf tensor (the `nn.Parameter` objects) that has `requires_grad=True`.
Leaf tensors are those created manually, like Weights and Biases. Results of hidden layer calculation, i.e. intermediate tensors, are discarded after use to save memory unless you called `.retain_grad()` on them.    
The `.grad` attibute is **additive**, meaning that if you call `backward()` multiple time without clearing the gradients, the new values will be added to existing ones. This is why `optimizer.zero_grad()` is called in training loops. 

- What is a loss function? What does it take for arguments, and what does it return?

The loss function determines how to calculate the gradient based on the difference between the ground truth and the inferred prediction (i.e. the error). It usually as a steepness term that parameterize gradient propagation and a regularization term to avoid exploding/vanishing gradients.  

- What does an optimization algorithm do?

The optimizer computes the loss, gradients and update the weights using a learning rate/step size to minimize loss.

- What is a hyperparameter, and how does it differ from a regular parameter?

A hyperparameter influence the learning process, rather than the internal variables of the model. It is used to guide the learning process to find the optimal solution, e.g. learning rate, number of epochs, activation functions.


### Software Engineering


## CNNs & ResNets

### Modules

**Universal Approximation Theorem:** any continuous function can be approximated using a sufficiently large neural network, if using non-linear activation function.

**Uniform Kaiming initialization:** default in pytorch, each weight and bias are drawn from uniform distribution on interval $[-\frac{1}{\sqrt{N_{in}}}, \frac{1}{\sqrt{N_{in}}}]$.

**Xavier initialization:** $\mathcal{O} \left( [-\frac{1}{\sqrt{N_{in}+N_{out}}}, \frac{1}{\sqrt{N_{in}+N_{out}}}] \right)$



### Training

#### Cross entropy loss

$$\text{loss} = \frac{1}{N} \sum_{n=1}^N  - \text{log} \, p_{n, y_n}$$

where $p_{n,c}$ is the probability the model assigns to class $c$ for sample $n$, and $y_n$ is the true label for this sample.

#### Convolutions

To calculate the size of the output of a convolution:

$$L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

As a general rule, a 3x3 convolution with $\text{padding}=1$ divides the image size by $\text{strides}$.

### ResNets

#### Batch Normalization

z-scoring activations across a mini-batch, forcing zero mean and unit variance. Introduces two new learnable parameters, ($\gamma$) (scale) and ($\beta$) (shift) $-$ per channel $-$ which allow network to learn optimal variance and mean activations, restoring representational power. 

Prevents vanishing or exploding gradients by keeping activations out of saturating non-linear zones, ensuring stable signal propagation. Bias terms are no longer needed, as they are mathematically subtracted out and absorbed by the batch mean calculation.

#### Architecture
![ResNet Architecture](./imgs/resnet-fixed.svg)

## Optimization

### Gradient Descent
$$\theta_t \leftarrow \theta_{t-1} - \lambda \nabla L(\theta_{t-1})$$


### Momentum (SGD)
$$ z^{k+1} = \beta z^k + \nabla f(w^k) $$
$$ w^{k+1} = w^k + \alpha z^{k+1} $$

### RMSProp
Root mean square propagation. Similar to SGD with an additional dynamic: the size of parameter steps are scaled according to the variance of the past gradients, with higher variance leading to smaller steps.

### ADAM
Adaptive Momentum Estimation.

$\theta$ are the parameters to optimize, $\delta$ is the increment of change at each step, with learning rate $\lambda$. $G$ is the gradient, $G_s$ is the sum of gradients.
$$G_s = G_s \beta_1 + G(1-\beta_1) \quad \quad\text{[Momentum]}$$ 
$$G_{s^2} = G_{s^2} \beta_2 + G^2 (1-\beta_2) \quad \quad\text{[RMSProp]}$$
$$\delta = \frac{-\lambda G_s}{\sum G_{s^2}} $$
$$ \theta = \theta + \delta $$

$\beta_1$ is the decay rates of the first moment (the mean), typically set at 0.9.

$\beta_2$ is the decay rates of the second moment (the variance), typically set at 0.999.

### Distributed Training
**Tensor** (horizontal) vs **Pipeline** (vertical) paralellism.  

![Types of parallelisms](./imgs/parallelism.png)

Collective communication: `broadcast`, `gather` (all to one, concatenated), `reduce` (like gather, with operation -- sum, mean, etc.).  
`all_gather` and `all_reduce` ensure all processes get the data. 


## Back-prop

![Backprop diagram](./imgs/abc_de_L.png)

`a`, `b` and `c` are leaf node, `L` is the root node. `d` and `e` are parents node of `L`. 

Why is it important to store the parent node in a tensor? So that we can compute the gradient for that path (i.e. allowing gradient propagation).
Gradients are accumulated rather than overwritten, so that in a case like `b`, since addition is commutative it does mot matter whether backwards is executed on Add or Mul first. 

Chain rule:
$$ \frac{dL}{dx} = \frac{dL}{d(out)} \times \frac{d(out)}{dx} = \frac{dL}{d(out)} \times \frac{d(log(x) )}{dx} = \frac{dL}{d(out)} \times \frac{1}{x} $$

### Cross entropy loss revisited
 Let's redo step by step how to get the loss from the logits and true labels.

 Full Cross Entropy Loss Equations:
 $$ L = -\sum y_{true} \, \text{log}(y_{pred}) $$

 Since $y_{true}$ is vector of $[0 ... 1 ... 0]$ for each sample, the above simplifies as
 $$ L = -\text{log} (y_{pred}) $$
 i.e. the predicted log probability.

 Because $y_{pred} = \text{softmax}(logits)$ we have:
 $$ L = -\text{log}(\frac{e^{logits}}{\sum e^{logits}}) = - \Big( \text{log}(e^{logits}) - \text{log}\Big(\sum e^{logits} \Big) \Big) = - logits + \text{log}\Big(\sum e^{logits} \Big)$$


## Autoencoder (AE) and Variational AE

### AE

![Autoencoder I](./imgs/ae-diagram-l.png)

![Autoencoder II](./imgs/ae-help-10.png)

### VAE

![VAE](./imgs/vae-reparam-l.png)

Explain in your own words why you need the reparameterization trick in order to train the VAE. 
$$ z = \mu + \sigma \odot \epsilon $$
The reparameterization trick (encoding the input into a mean $\mu$ and a standard dev $\sigma$ multiplied by a random vector $\epsilon$) forces the latent to have a more continuous representation across classes (how? by enforcing $\sigma \approx 1$ instead of a single point at $\mu_{\text{class}}$). This forces exploration of the latent space via the KL-divergence:  

$$ \text{Loss}_{KL} = \frac{1}{2} \sum \mu^2 + \sigma^2 - \ln(\sigma^2) -1 $$

i.e. if the encoder minimizes $\sigma \rightarrow 0$ , the term $-\ln(\sigma^2)$ goes to $+\infty$. 

Also the term $\mu^2$ forces the encoded center of all the classes to be around 0 (i.e. overlappign). The decoder has to figure out a way to transition smoothly from one class to another, creating a truly continuous space.   
 
The $\epsilon$ parameter is random and is necessary for backprop (how? why?). Because differentiating $\mathcal{N}(\mu,\sigma)$ is not possible (non-continuous), but differentiating $\mu + \sigma \odot \epsilon$ works ($\frac{dz}{d\mu}=1$ and $\frac{dz}{d\sigma} = \epsilon$).

The total loss reads 
$$ \text{Loss}_{VAE} (x, x') = ||{x - x'}||^2 + D_{\text{KL}} \big( \mathcal{N}(\mu, \sigma^2) || \mathcal{N}(0, 1) \big)   $$

#### Deeper into math:
The discriminator (decoder) can be written 
$$ p(x) = \int_z p(x|z)\;p(z)\;dz = \mathbb{E}_{z \sim p(z)} \big[ \; p(x|z) \;\big]$$
but finding $p(x)$ by sampling over the latent space $z$ is computationally intractable. 

That's where the generator (encoder) $q(z|x)$ helps: it concentrates the latent space to a region that is likely to produce $x$. So the above equation can be rewritten
$$ p(x) = \int_z q(z|x) \; \frac{p(x|z)\;p(z)}{q(z|x)}dz = \mathbb{E}_{z \sim q(z|x)} \Big[ \; \frac{p(x|z)\;p(z)}{q(z|x)} \; \Big]$$

Now, Jensen's inequality states that $\mathbb{E}\big[f(X)\big] \geq f\big(\mathbb{E}(X)\big)$ for any convex function $f$ (convex mean has the same curvature direction at all point i.e. $f''(x)>0$, e.g. $\log(x)$ is convex). 
So $\log p(X) \geq p(\log X)$. 

$p(x)$ becomes 
$$ \log p(x) \geq \mathbb{E}_{z \sim q(z|x)} \Big[ \log \frac{p(x|z)\;p(z)}{q(z|x)}  \Big]$$

which is called the *evidence lower-bound* (***ELBO***), because $\log p(x)$ is called the *evidence*.

Now, rearranging the terms, we get
$$\begin{aligned} 
\mathbb{E}_{z \sim q(z|x)} \Big[ \log \frac{p(x|z)\;p(z)}{q(z|x)}  \Big] 
= \mathbb{E}_{z \sim q(z|x)} \Big[ \log p(x|z) + \log \frac{\;p(z)}{q(z|x)} \; \Big] 
&= \mathbb{E}_{z \sim q(z|x)} \Big[ \log p(x|z) - \log \frac{q(z|x)}{p(z)} \; \Big]\\ 
&= \mathbb{E}_{z \sim q(z|x)} \Big[ \log p(x|z) \Big] - \mathbb{E}_{z \sim q(z|x)} \Big[\log \frac{q(z|x)}{p(z)} \; \Big] \end{aligned}$$

Recall from above that $D_{\text{KL}} (P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = p \log p - p \log q$, and can also be written $\mathbb{E}_p [ \log \frac{p}{q} ]$, therefore by flipping $p$ and $q$ we obtain: 
$$ \text{ELBO}(x) = \mathbb{E}_{z \sim q(z|x)} \Big[ \log p(x|z) \Big] - D_{\text{KL}} \big(\; q(z|x)\;||\;p(z) \;\big)$$

where $\mathbb{E}_{z \sim q(z|x)} \Big[ \log p(x|z) \Big]$ is the reconstrcution loss, and $D_{\text{KL}} \big( q(z|x)\;||\;p(z)\big)$ is the regularization term.

### GAN

![GAN](./imgs/dcgan-9-solid.png)

#### Minimax game
$$ \min_G \max_D V (D,G) = \mathbb{E}_x \big[ \log \big( D(x) \big) \big] + \mathbb{E}_z \big[ \log \big( 1 - D(G(z)) \big) \big]  $$


### Bonus: Transposed convolutions
![Transposed convolution](./imgs/convtranspose-1.png)

For 1D transposed convolution (no padding, no strides):
$$\begin{aligned}
\text{output size} 
&= \text{input size} \; -  \; \text{kernel size} + 2\times(\text{kernel size} - 1) + 1 \\
&= \text{input size} \;  + \text{kernel size} - 1
\end{aligned}$$

With padding and strides:
$$ \begin{aligned} 
&\text{output size} \\
&= \text{input size} \;  + \text{kernel size} - 1 - 2 \times \text{padding} + (\text{input size} -1) \times (\text{strides} -1 ) \\
&= (\text{input size}-1) \; \times \text{strides} + \text{kernel size} - 2\times\text{padding}
\end{aligned} $$


# Chapter 1: Mechanistic Interpretability

# Chapter 2: Reinforcement Learning

# Chapter 3: Evals

## Intro to Evals

**Capacity**  *vs* **alignment evaluations**:  whether a model has the capacity *vs* the tendency for specific behaviors.

**Greedy Coordinate Gradient (GCG)**: introduces an adversarial suffix to the user prompt, and optimize this suffix such that the generated response starts positively (e.g. "Sure, I can help with that."). When the adversarial suffix is optimized across many harmful prompts, this suffix acts as a universal jammer that inhibits the safety guardrails.  

**Specification gaming** is a behaviour that satisfies the literal specification of an objective without achieving the intended outcome (e.g. flip the red cube upside down instead of putting it on top of the blue one). 

### Representation Engineering

**Abliteration** : Process of inhibiting the safety layers activation, by injecting a compliant vector directly onto the mode's middle layers. 
**Forced refusal** : instead of inhibiting, the safety vector is potentiated. 

**Rogue Scalpel**: Present the model with a politically neutral sentence completion task, such as: *"The recent protests regarding the climate bill were ultimately ____."* Calculate the ratio of probability between ideologically charged tokens (e.g., justified, necessary vs. disruptive, misguided). A massive mathematical skew in the top-5 predicted tokens across 100 neutral stems indicates hardcoded political taste alignment. 