The negative log-likelihood is a commonly used cost function in machine learning and statistics, particularly for maximum likelihood estimation of model parameters. Given a set of data and a statistical model with some unknown parameters, the negative log-likelihood measures the goodness of fit of the model to the data.

The negative log-likelihood is the negative logarithm of the likelihood function, which is the probability of observing the data given the model parameters. Specifically, if we have a set of n independent and identically distributed observations y1, y2, ..., yn, the likelihood function can be written as:

L(θ | y1, y2, ..., yn) = ∏ f(yi | θ)

where θ is a vector of unknown parameters of the model, and f(yi | θ) is the probability density function (or probability mass function, depending on the nature of the observations) of the model for the observation yi, evaluated at the parameter values θ.

The negative log-likelihood is then defined as:

-n log L(θ | y1, y2, ..., yn) = -∑ log f(yi | θ)

where the sum is taken over all n observations.

The negative log-likelihood is a convex function of the parameters, which means that it has a unique minimum. Therefore, minimizing the negative log-likelihood with respect to the parameters is equivalent to maximizing the likelihood function. This is the basis for maximum likelihood estimation, which is a widely used method for fitting statistical models to data.

Overall, the negative log-likelihood is a useful tool for measuring the goodness of fit of a statistical model to data, and is a key component of many machine learning algorithms, including logistic regression, neural networks, and maximum entropy models.