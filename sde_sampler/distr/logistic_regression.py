# Horseshoe Logistic Regression

# Libraries
import pickle
import torch
from torch.distributions.utils import probs_to_logits
from torch.nn.functional import binary_cross_entropy_with_logits
from .base import DATA_DIR, Distribution


class LogisticRegression(Distribution):
    """Base class for Logistic Regression distribution."""

    def __init__(self, dim, data_type, use_intercept=True, intercept_mean=0.0, intercept_scale=2.5, weight_scale=1.0, threshold=1e-8, **kwargs):
        # Dataset
        with open(DATA_DIR / "{}.pkl".format(data_type), 'rb') as f:
            data = pickle.load(f)
        self.X_train = data['X_train'].float()
        self.y_train = data['y_train'].float().flatten()
        self.X_test = data['X_test'].float()
        self.y_test = data['y_test'].float().flatten()
        dim_weights = self.X_train.shape[-1]
        super().__init__(dim=dim_weights + int(use_intercept), **kwargs)
        # Priors
        self.threshold = 1e-8
        self.register_buffer("weight_scale", torch.tensor(weight_scale), persistent=False)
        self.weights_prior = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(
                loc=torch.zeros((dim_weights,)),
                scale=self.weight_scale * torch.ones((dim_weights,))
            ), reinterpreted_batch_ndims=1)
        self.use_intercept = use_intercept
        if use_intercept:
            self.register_buffer("intercept_mean", torch.tensor(intercept_mean), persistent=False)
            self.register_buffer("intercept_scale", torch.tensor(intercept_scale), persistent=False)
            self.intercept_prior = torch.distributions.Normal(
                loc=self.intercept_mean,
                scale=self.intercept_scale
            )

    def posterior_log_prob(self, params, X, y):
        """Evaluates the posterior density at params, conditioned on (X,y)"""
        # Ensure the shape of the params
        params = params.reshape((-1, params.shape[-1]))
        # Unpack the parameters
        if self.use_intercept:
            weights, intercept = params[..., :-1], params[..., -1]
        else:
            weights = params
        # Compute the prior
        prior_log_prob = self.weights_prior.log_prob(weights)
        if self.use_intercept:
            prior_log_prob += self.intercept_prior.log_prob(intercept)
        # Compute the likelihood
        probs = torch.special.expit(torch.matmul(X, weights.T).T + intercept.unsqueeze(-1))
        probs = torch.clip(probs, self.threshold, 1.0 - self.threshold)
        logits = probs_to_logits(probs, is_binary=True)
        log_prob = - \
            binary_cross_entropy_with_logits(logits, y.unsqueeze(
                0).expand((logits.shape[0], -1)), reduction="none").sum(dim=-1)
        return log_prob + prior_log_prob

    def posterior_score(self, params, X, y):
        """Evaluates the score of the posterior distribution at params, conditioned on (X,y)"""
        # Ensure the shape of the params
        params = params.reshape((-1, params.shape[-1]))
        # Unpack the parameters
        if self.use_intercept:
            weights, intercept = params[..., :-1], params[..., -1]
        else:
            weights = params
        # Get the score of the priors
        prior_weights_score = -weights / self.weight_scale**2
        if self.use_intercept:
            prior_intercept_score = -(intercept.unsqueeze(-1) - self.intercept_mean) / self.intercept_scale**2
        # Compute the probs
        probs = torch.special.expit(torch.matmul(X, weights.T).T + intercept.unsqueeze(-1))
        probs = torch.clip(probs, self.threshold, 1.0 - self.threshold)
        # Get the score with respect to the weights
        prior_weights_score += torch.einsum('bn,nd->bd', y.unsqueeze(0) - probs, X)
        if self.use_intercept:
            prior_intercept_score += torch.sum(y.unsqueeze(0) - probs, dim=-1, keepdim=True)
            return torch.concat([prior_weights_score, prior_intercept_score], dim=-1)
        else:
            return prior_weights_score

    def unnorm_log_prob(self, x, *args, **kwargs):
        """Evaluates the unnormalized posterior density at params, conditioned on (X_train,y_train)"""
        return self.posterior_log_prob(x, self.X_train, self.y_train).unsqueeze(-1)

    # def score(self, x, *args, **kwargs):
    #     return self.posterior_score(x, self.X_train, self.y_train)

    def compute_predictive_log_prob(self, x):
        """Evaluates the unnormalized posterior density at params, conditioned on (X_test,y_test)"""
        return self.posterior_log_prob(x, self.X_test, self.y_test).mean()

    def _apply(self, fn):
        super(LogisticRegression, self)._apply(fn)
        self.X_train = fn(self.X_train)
        self.X_test = fn(self.X_test)
        self.y_train = fn(self.y_train)
        self.y_test = fn(self.y_test)
        self.weights_prior.base_dist.loc = fn(self.weights_prior.base_dist.loc)
        self.weights_prior.base_dist.scale = fn(self.weights_prior.base_dist.scale)
        self.intercept_prior.loc = fn(self.intercept_prior.loc)
        self.intercept_prior.scale = fn(self.intercept_prior.scale)
