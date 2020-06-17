# Built upon https://github.com/ikostrikov/pytorch-flows/

import collections.abc

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import amci.utils

EPS = 1e-7


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=bias)

        self.register_buffer('mask', mask)

    def forward(self, input, cond_inputs=None):
        output = F.linear(input, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class ConditionedMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super().__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   MaskedLinear(num_hidden, num_hidden, hidden_mask),
                                   act_func(),
                                   MaskedLinear(num_hidden, num_inputs * 2, output_mask))

    def forward(self, input, mode='forward', conditioned_on=None):
        if mode == 'forward':
            x = torch.zeros_like(input)
            for i_col in range(input.shape[1]):
                h = self.joiner(x, conditioned_on)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = input[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, a.sum(-1)
        else:
            h = self.joiner(input, conditioned_on)
            m, a = self.trunk(h).chunk(2, 1)
            u = (input - m) * torch.exp(-a)
            return u, -a.sum(-1)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, input, mode='forward', conditioned_on=None):
        if mode == 'forward':
            return input[:, self.perm], torch.zeros(input.shape[0])
        else:
            return input[:, self.inv_perm], torch.zeros(input.shape[0])


class RadialFlow(nn.Module):
    """ An implementation of the radial layer from
    Variational Inference with Normalizing Flows
    (https://arxiv.org/abs/1505.05770).
    """

    def __init__(self, d):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(1, d))
        self.log_a = nn.Parameter(torch.zeros(1, 1))
        # bhat is b before reparametrization
        self.bhat = nn.Parameter(torch.zeros(1, 1))

        self.d = d

    def forward(self, input, mode='forward'):
        # print('RadialFlow', self.z0, self.log_a, self.bhat)

        assert input.shape[1] == self.d

        # Offset to make the flow an identity flow if all parameters are zeros.
        #  Initializing the parameter to this value doesn't do the job when we're using a hyper-network.
        bhat = self.bhat + (torch.ones(1, 1).exp() - torch.ones(1, 1)).log()

        d = float(self.d)
        a = torch.exp(self.log_a)
        # According to the Appendix in the paper
        b = -a + torch.nn.functional.softplus(bhat)
        if mode == 'forward':
            z = input
            z_z0 = z - self.z0
            r = z_z0.norm(dim=1, keepdim=True) + EPS
            h = 1. / (a + r)
            hprime = -1. / (a + r).pow(2.)
            logdet = (d - 1.) * (1. + b * h).log() + (1. + b * h + b * hprime * r).log()
            y = z + b * z_z0 * h
            return y, logdet.reshape(-1)
        else:
            y = input
            y_z0 = y - self.z0
            c = y_z0.norm(dim=1, keepdim=True)
            B = a + b - c
            sqrt_delta = (B.pow(2.) + 4. * a * c).sqrt()
            r = 0.5 * (-B + sqrt_delta) + EPS
            h = 1. / (a + r)
            zhat = y_z0 / (r * (1. + b / (a + r)))
            hprime = -1. / (a + r).pow(2)
            z = self.z0 + r * zhat
            inv_logdet = -((d - 1.) * (1. + b * h).log() + (1. + b * h + b * hprime * r).log())
            return z, inv_logdet.reshape(-1)


class OffsetFlow(nn.Module):
    """ A flow layer adding an offset. """

    def __init__(self, d):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, d))
        self.d = d

    def forward(self, input, mode='forward', params=None, **kwargs):
        # print('OffsetFlow', self.loc)
        assert input.shape[1] == self.d
        if mode == 'forward':
            x = input
            y = self.loc + x
            logdet = torch.zeros(input.shape[0])
            return y, logdet
        else:
            y = input
            x = y - self.loc
            inv_logdet = torch.zeros(input.shape[0])
            return x, inv_logdet


class FlowDensityEstimator(nn.Module):
    """
    A density estimator for a sequance of flow modules.
    It is able to evaluate log_prob and generate samples.
    Requires specifying a base distribution.
    Implements both forward and backward passes and computes log jacobians.
    """

    def __init__(self, base_distribution, flow_modules):
        super().__init__()

        self.flow = nn.ModuleList(flow_modules)

        if isinstance(base_distribution, torch.distributions.Distribution):
            self.base_distribution = self.ThinTorchDistributionWrapper(base_distribution)
        elif isinstance(base_distribution, LearnableDistribution):
            self.base_distribution = base_distribution
        else:
            raise Exception('base_distribution takes an object of class amci.flows.LearnableDistribution or '
                            'torch.distributions.Distribution')

    def forward(self, input, function='log_prob', return_logprobs=False):
        """
        Necessary for FlowDensityEstimator to be used in ConditionedDensityEstimator.
        """
        if function == 'log_prob':
            return self.log_prob(input)
        elif function == 'sample':
            return self.sample(input, return_logprobs=return_logprobs)
        else:
            raise Exception(f'Unknown value of argument function: {function}')

    def sample(self, batch_size=None, return_logprobs=False, conditioned_on=None):
        """
        Returns samples (and optionally their log_probs) from the distribution.
        The samples tensor is of shape [batch_size, d] where d depends on the dimensionality of the base distribution and the flow modules provided to the constructor.
        See tests/test_flows.py to see the interface in examples.
        :param batch_size: int
        :param return_logprobs: bool, if true returns a tuple (samples, log_prob) rather than just a single variable samples
        :param conditioned_on: when using flow consisting of ConditionedMADE need to pass conditioning tensors of shape.
        :return: samples or (samples, log_probs)
        """
        params_shape = self.parameters().__next__().shape[0]
        if batch_size is not None:
            assert isinstance(batch_size, int)
        elif not any(isinstance(module, ConditionedMADE) for module in self.flow):
            batch_size = params_shape
        else:
            batch_size = 1

        if (not (batch_size == params_shape or params_shape == 1 or batch_size == -1)) and \
                not any(isinstance(module, ConditionedMADE) for module in self.flow):
            raise ValueError("batch_size doesn't match the batch size of the parameters.")

        x = self.base_distribution(function='sample', input=batch_size)
        y, logdets = self.flow_pass(x, mode='forward', conditioned_on=conditioned_on)
        if return_logprobs:
            base_log_prob = self.base_distribution(function='log_prob', input=x)
            log_prob = base_log_prob - logdets
            return y, log_prob
        else:
            return y

    def log_prob(self, value, conditioned_on=None):
        """
        Returns the log_probs of the `value` according to the distribution.
        See tests/test_flows.py to see the interface in examples.
        :param value:
        :param conditioned_on: when using flow consisting of ConditionedMADE need to pass conditioning tensors of shape.
        :return: log_probs
        """
        x, invlogdets = self.flow_pass(value, mode='reverse', conditioned_on=conditioned_on)
        assert len(invlogdets.shape) == 1
        base_log_prob = self.base_distribution(function='log_prob', input=x)
        assert len(base_log_prob.shape) == 1
        log_prob = base_log_prob + invlogdets
        return log_prob

    def flow_pass(self, inputs, mode='forward', conditioned_on=None):
        """
        Performs a forward or backward pass for flow modules.
        :param inputs: inputs to the flow
        :param mode: str, 'forward' or 'reverse', the direction of the computation of the flow
            convention in this codebase is that 'reverse' is for log_prob, 'forward' is for sampling
        :param conditioned_on: when using flow consisting of ConditionedMADE need to pass conditioning tensors
        """
        logdets = torch.zeros(inputs.shape[0])

        assert mode in ['forward', 'reverse']
        if mode == 'forward':
            modules_list = self.flow
        else:
            modules_list = reversed(self.flow)
        if conditioned_on is not None:
            assert isinstance(conditioned_on, collections.abc.Sequence)
            conditioned_on = amci.utils.cat_last_dim(conditioned_on, with_expand=True)
        for module in modules_list:
            if conditioned_on is not None:
                inputs, logdet = module(inputs, mode, conditioned_on=conditioned_on)
            else:
                inputs, logdet = module(inputs, mode)
            logdets = logdets + logdet

        return inputs, logdets

    class ThinTorchDistributionWrapper(nn.Module):
        """
        A thin wrapper for base distributions specified using torch.distributions.Distribution objects.
        Required for compatibility with the rest of the APIs in this module.
        """
        def __init__(self, distribution):
            super().__init__()
            self.distribution = distribution

        def forward(self, input, function='log_prob'):
            if function == 'log_prob':
                return self.distribution.log_prob(input)
            elif function == 'sample':
                if input is None:
                    batch_size = 1
                else:
                    batch_size = input
                return self.distribution.sample(sample_shape=(batch_size,))
            else:
                raise Exception(f'Unknown value of argument function: {function}')


class ConditionedDensityEstimator:
    """
    A conditioned density estimator which determines the parameters of the underlying normalizing flow based on the
    conditioning values using a hyper-network.
    Flows consisting of ConditionedMADE do not need to use ConditionedDensityEstimator and a hyper-network because the
    conditioning mechanism is incorporated into the logic of the flow itself.
    See tests/test_flows.py to see the interface in examples.
    """
    def __init__(self, hyper_net, density_estimator):
        self.hyper_net_module = hyper_net
        # the first dimension is the batch dimension that should be 1 so we skip it in calculating the total number
        #  of parameters in the model
        self.density_estimator_parameters_shapes = tuple(param.shape[1:] for param in density_estimator.parameters())
        # functional (stateless) module
        self.density_estimator = higher.patch.monkeypatch(density_estimator, track_higher_grads=False)

    def density_estimator_params(self, conditioned_on):
        assert isinstance(conditioned_on, collections.abc.Sequence)
        conditioned_on = amci.utils.cat_last_dim(conditioned_on, with_expand=True)
        params = self.hyper_net_module(conditioned_on)
        params = torch.split(params,
                             tuple(np.prod(param_shape) for param_shape in self.density_estimator_parameters_shapes),
                             dim=-1)
        params = tuple(param_values.reshape(conditioned_on.shape[0], *param_shape)
                       for param_values, param_shape in zip(params, self.density_estimator_parameters_shapes))
        return params

    def log_prob(self, value, conditioned_on=tuple()):
        return self.density_estimator(value, function="log_prob",
                                      params=self.density_estimator_params(conditioned_on))

    def sample(self, batch_size=None, conditioned_on=tuple(), return_logprobs=False):
        return self.density_estimator(batch_size, function="sample", return_logprobs=return_logprobs,
                                      params=self.density_estimator_params(conditioned_on))

    def parameters(self):
        return self.hyper_net_module.parameters()

    def state_dict(self):
        return self.hyper_net_module.state_dict()

    def load_state_dict(self, state_dict):
        return self.hyper_net_module.load_state_dict(state_dict)


class LearnableDistribution(nn.Module):
    def distribution(self, dist_params, batch_size=None):
        if batch_size is None:
            batch_size = -1

        if isinstance(dist_params, dict):
            params_shape = tuple(dist_params.values())[0].shape[0]
        else:
            params_shape = dist_params[0].shape[0]

        if not (batch_size == params_shape
                or params_shape == 1
                or batch_size == -1):
            raise ValueError("batch_size doesn't match the batch size of the parameters.")

        if isinstance(dist_params, dict):
            dist_params = {k: v.expand(batch_size, -1) for k, v in dist_params.items()}
            return self.dist_class(**dist_params)
        else:
            dist_params = tuple(v.expand(batch_size, -1) for v in dist_params)
            return self.dist_class(*dist_params)

    def forward(self, input, function, return_logprobs=False):
        if function == 'log_prob':
            return self.distribution(self.distribution_parameters()).log_prob(input)
        elif function == 'sample':
            if input:
                assert isinstance(input, int)
                assert input > 0
            distribution = self.distribution(self.distribution_parameters(), batch_size=input)
            samples = distribution.sample()
            if not return_logprobs:
                return samples
            else:
                return samples, distribution.log_prob(samples)
        else:
            raise Exception(f'Unknown value of argument function: {function}')

    def sample(self, batch_size=None, return_logprobs=False):
        return self(function='sample', input=batch_size, return_logprobs=return_logprobs)

    def log_prob(self, value):
        return self(function='log_prob', input=value)


class Gamma(LearnableDistribution):
    def __init__(self, d=1):
        super().__init__()
        self.dist_class = lambda *args, **kwargs: \
            torch.distributions.independent.Independent(
                torch.distributions.Gamma(*args, **kwargs), 1)
        self.presoftplus_mean = nn.Parameter(torch.zeros(1, d))
        self.presoftplus_std = nn.Parameter(torch.zeros(1, d))

    def distribution_parameters(self):
        mean = F.softplus(self.presoftplus_mean)
        std = F.softplus(self.presoftplus_std)
        param0 = mean ** 2 / std ** 2
        param1 = mean / std ** 2
        return param0, param1


class Beta(LearnableDistribution):
    def __init__(self, d=1):
        super().__init__()
        self.dist_class = lambda *args, **kwargs: \
            torch.distributions.independent.Independent(
                torch.distributions.Beta(*args, **kwargs), 1)
        self.params = nn.Parameter(torch.zeros(1, 2 * d))
        self.d = d

    def distribution_parameters(self):
        c0, c1 = self.params.split(self.d, dim=1)
        c0 = F.softplus(c0)
        c1 = F.softplus(c1)
        return c0, c1
