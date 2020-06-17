import pytest
import torch

import amci.flows
import amci.utils

N = 2


def get_flow(type, d):
    if type == 'radial':
        return amci.flows.RadialFlow(d=d)
    elif type == 'conditioned_made':
        return amci.flows.ConditionedMADE(num_hidden=10, num_inputs=d, num_cond_inputs=1)


@pytest.mark.parametrize(("flow_type", "d"),
                         (('radial', 1), ('radial', 5), ('conditioned_made', 5)))
def test_flow_invertibility(flow_type, d):
    flow = get_flow(flow_type, d)

    for param in flow.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)

    x = torch.randn((N, d))
    y, logdets = flow(x, mode='forward')
    x_recon, invlogdets = flow(y, mode='reverse')

    assert torch.allclose(x, x_recon)
    assert torch.allclose(logdets, -invlogdets)


@pytest.mark.parametrize(("flow_type", "d"),
                         (('radial', 1), ('radial', 5), ('conditioned_made', 5)))
def test_flow_sample_log_prob(flow_type, d):
    flow_modules = tuple(get_flow(flow_type, d) for _ in range(5))
    base_distribution = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)
    density_esimator = amci.flows.FlowDensityEstimator(base_distribution=base_distribution,
                                                       flow_modules=flow_modules)

    for param in density_esimator.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)

    y, y_sample_log_prob = density_esimator.sample(N, return_logprobs=True)
    y_log_prob = density_esimator.log_prob(y)

    assert torch.allclose(y_sample_log_prob, y_log_prob)


def get_density_estimator(type, d):
    if type == 'gamma':
        density_esimator = amci.flows.Gamma(d=d)
    elif type == 'beta':
        density_esimator = amci.flows.Beta(d=d)

    if type[:4] == 'flow':
        if type == 'flow_radial':
            flow_modules = tuple((amci.flows.RadialFlow(d=d), amci.flows.RadialFlow(d=d), amci.flows.OffsetFlow(d)))
            base_distribution = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)
        elif type == 'flow_with_learnable_base':
            flow_modules = tuple((amci.flows.RadialFlow(d=d), amci.flows.RadialFlow(d=d), amci.flows.OffsetFlow(d)))
            base_distribution = amci.flows.Gamma(d=d)
        elif type == 'flow_conditioned_made':
            flow_modules = (amci.flows.ConditionedMADE(num_inputs=d, num_hidden=10, num_cond_inputs=1),
                            amci.flows.Reverse(d),
                            amci.flows.ConditionedMADE(num_inputs=d, num_hidden=10, num_cond_inputs=1))
            base_distribution = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)

        density_esimator = amci.flows.FlowDensityEstimator(base_distribution=base_distribution,
                                                           flow_modules=flow_modules)

    return density_esimator


@pytest.mark.parametrize("density_estimator_type",
                         ('beta', 'gamma', 'flow_radial', 'flow_with_learnable_base', 'flow_conditioned_made'))
@pytest.mark.parametrize("d", (5,))
def test_density_estimator_shapes(density_estimator_type, d):
    de = get_density_estimator(density_estimator_type, d)

    y = de.sample(return_logprobs=False)
    assert y.shape == (1, d)
    y, y_sample_log_prob = de.sample(return_logprobs=True)
    assert y.shape == (1, d)
    assert y_sample_log_prob.shape == (1,)

    y = de.sample(N, return_logprobs=False)
    assert y.shape == (N, d)
    y, y_sample_log_prob = de.sample(N, return_logprobs=True)
    assert y.shape == (N, d)
    assert y_sample_log_prob.shape == (N,)

    with pytest.raises(Exception):
        de.sample((1,), return_logprobs=False)
    with pytest.raises(Exception):
        de.sample((N,), return_logprobs=False)
    with pytest.raises(Exception):
        de.sample((N, 1), return_logprobs=False)


@pytest.mark.parametrize("density_estimator_type",
                         ('beta', 'gamma', 'flow_radial', 'flow_with_learnable_base'))
@pytest.mark.parametrize("d", (5,))
def test_conditioned_density_estimator_shapes(density_estimator_type, d):
    density_estimator = get_density_estimator(density_estimator_type, d)
    parameters_nelement = sum(param.nelement() for param in density_estimator.parameters())
    hyper_net = amci.utils.get_flow_hyper_net(
        hidden_units_per_layer=2, parameters_nelement=parameters_nelement, in_dim=1, number_of_layers=2)
    cde = amci.flows.ConditionedDensityEstimator(
        hyper_net=hyper_net, density_estimator=density_estimator)

    conditioned_on = (torch.randn((1, 1)),)
    assert cde.sample(conditioned_on=conditioned_on).shape == (1, d)
    assert cde.sample(1, conditioned_on=conditioned_on).shape == (1, d)
    assert cde.sample(N, conditioned_on=conditioned_on).shape == (N, d)

    conditioned_on = (torch.randn((N, 1)),)
    assert cde.sample(conditioned_on=conditioned_on).shape == (N, d)
    assert cde.sample(N, conditioned_on=conditioned_on).shape == (N, d)
    # batch_size must be matching the batch_size
    with pytest.raises(Exception):
        cde.sample(1, conditioned_on=conditioned_on)

    with pytest.raises(Exception):
        cde.sample((1,), return_logprobs=False)
    with pytest.raises(Exception):
        cde.sample((N,), return_logprobs=False)
    with pytest.raises(Exception):
        cde.sample((N, 1), return_logprobs=False)


if __name__ == '__main__':
    pytest.main()
