# Toeplitz-Covariance-Estimation-via-Overparametrized-Gradient-Descent

Code of the paper "Toeplitz Covariance Estimation via Overparametrized Gradient Descent",
By Daniel Busbib and Ami Wiesel

## Abstract

We consider covariance estimation under Toeplitz structure. Numerous sophisticated optimization methods have been
developed to maximize the Gaussian log-likelihood under Toeplitz constraints. In contrast, recent advances in deep
learning demonstrate the surprising power of simple gradient descent (GD) applied to overparameterized models. Motivated
by this trend, we revisit Toeplitz covariance estimation through the lens of overparameterized GD. We model the
covariance as a sum of K complex sinusoids with learnable parameters and optimize them via GD. We show that when K =
P (the matrix dimension), GD may converge to suboptimal solutions. However, mild overparameterization (K = 2P or
4P) consistently enables global convergence from random initializations. Our experiments demonstrate that
overparameterized GD can match or exceed the accuracy of state-of-the-art methods in challenging settings, while
remaining simple and scalable.

## Citation

coming soon
