import numpy as np
import scipy.stats
import torch

import amci.flows as flows
import amci.utils


class Training:
    def __init__(self, q1_or_q2, q, args):
        assert args.tail_integral_d in [1, 5], "d should be either 1 or 5."
        self.d = args.tail_integral_d

        if q1_or_q2 not in ['q1', 'q2']:
            raise Exception(f'Unrecognized value of the argument q1_or_q2: {q1_or_q2}.')
        self.q1_or_q2 = q1_or_q2

        self.q = q

        self.args = args

        self.init_distributions()

    def init_distributions(self):
        d = self.d

        # This is used as p(x) and for the p(y|x) = N(x, 1) => y = x + N(0, 1)
        # torch.distributions.Independent(p, 1) reinterprets the first batch dim of a distribution p as an event dim.
        self.standard_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)

        self.p_theta = torch.distributions.Independent(
            torch.distributions.Uniform(0., self.args.max_theta * torch.ones(d)), 1)

        # self.q_prime_x is the proposal distribution for training the proposal q1, see Eq. 22 in the AMCI paper
        if d == 1:
            self.p_x = self.standard_normal
            self.q_prime_x = torch.distributions.Independent(
                torch.distributions.HalfNormal(torch.ones(d)), 1)
        elif d == 5:
            self.p_x = torch.distributions.MultivariateNormal(torch.zeros(d), self.covariance_matrix_5d(d))
            self.q_prime_x = torch.distributions.Independent(
                torch.distributions.HalfNormal(torch.diag(self.covariance_matrix_5d(d))), 1)

    @staticmethod
    def get_proposal_model(args):
        """
        Returns the proposal distribution q.
        """

        d = args.tail_integral_d

        # specification of the flow
        if args.q1_or_q2 == 'q1':
            hyper_net_in_dim = 2 * d
            flow_layers = args.layers_q1
        elif args.q1_or_q2 == 'q2':
            hyper_net_in_dim = d
            flow_layers = args.layers_q2
        else:
            raise Exception(f'Unrecognized value of the argument q1_or_q2: {args.q1_or_q2}.')

        flow_modules = []
        if d == 1:
            flow_modules += [flows.RadialFlow(d) for _ in range(flow_layers)]
            if args.q1_or_q2 == 'q1':
                flow_modules += [flows.OffsetFlow(d)]
            base_distribution = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)

            density_esimator = flows.FlowDensityEstimator(base_distribution, flow_modules)

            # the first dimension is the batch dimension that should be 1 so we skip it in calculating the total number
            #  of parameters in the model
            parameters_nelement = sum(np.prod(param.shape[1:]) for param in density_esimator.parameters())

            hyper_net = amci.utils.get_flow_hyper_net(args.hidden_units_per_layer,
                                                      parameters_nelement,
                                                      hyper_net_in_dim)

            q = amci.flows.ConditionedDensityEstimator(hyper_net=hyper_net, density_estimator=density_esimator)
        elif d == 5:
            for _ in range(flow_layers):
                flow_modules += [flows.ConditionedMADE(d, args.MADE_hidden, hyper_net_in_dim, act=args.MADE_act)]
                flow_modules += [flows.Reverse(d)]

            base_distribution = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros(d), torch.ones(d)), 1)
            q = flows.FlowDensityEstimator(base_distribution, flow_modules)
        else:
            raise Exception(f'Unrecognized value of the argument tail_integral_d: {args.tail_integral_d}.')

        return q

    @staticmethod
    def covariance_matrix_5d(d):
        """ Covariance matrix for 5d tail integral p(x) prior. """
        a = np.tril(np.linspace(0.20, .25, d * d).reshape(d, d)).T
        sigma = a @ a.T + np.eye(d)
        return torch.tensor(sigma, dtype=torch.float)

    @staticmethod
    def f(x, theta):
        return ((x > theta).sum(dim=-1) == x.shape[-1]).to(torch.float).reshape(-1)

    def generate_dataset(self, dataset_size):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, size of the dataset to be generated
        :return: the dataset in a form of a dict.
        """
        if self.q1_or_q2 == 'q1':
            theta = self.p_theta.sample((dataset_size,))

            x = self.q_prime_x.sample((dataset_size,))
            assert theta.shape == x.shape
            x += theta
            y = x + self.standard_normal.sample((dataset_size,))

            # we will denote the evaluated values of the function as f_x
            f_x = self.f(x, theta)

            return {'x': x, 'y': y, 'f_x': f_x, 'theta': theta}

        elif self.q1_or_q2 == 'q2':
            x = self.p_x.sample((dataset_size,))
            y = x + self.standard_normal.sample((dataset_size,))

            return {'x': x, 'y': y}

    def loss(self, x=None, y=None, theta=None, f_x=None):
        """
        Computes the loss function.
        """

        if self.q1_or_q2 == 'q1':
            # Eq. 22
            loss = -((self.p_x.log_prob(x) - self.q_prime_x.log_prob(x - theta)).exp() *
                     f_x * self.q.log_prob(x, conditioned_on=(y, theta)))
        elif self.q1_or_q2 == 'q2':
            # Eq. 7
            loss = -self.q.log_prob(x, conditioned_on=(y,))

        return loss


class Evaluation(Training):
    def __init__(self, args, load_checkpoints=True):
        # This is used as p(x) and for the p(y|x) = N(x, 1) => y = x + N(0, 1)
        d = args.tail_integral_d
        self.d = d

        self.args = args

        self.init_distributions()

        if load_checkpoints:
            model_parameters_names = ('max_theta',)
            self.loaded_checkpoint_parameters = \
                amci.utils.load_proposals_from_checkpoints(self, ('q1', 'q2'), model_parameters_names)

    def logw_for_groundtruth(self, N, y_theta):
        y, theta = (y_theta[k] for k in ['y', 'theta'])

        x_minus_theta = self.q_prime_x.sample((N,)).unsqueeze(0)
        x = x_minus_theta + theta.unsqueeze(1)
        # all of the samples x yield f(x)==1 by construction (because we add theta)
        #  so we don't need to call the function f
        logw_numerator = self.standard_normal.log_prob(y.unsqueeze(1) - x) + \
                         self.p_x.log_prob(x) - \
                         self.q_prime_x.log_prob(x_minus_theta)

        x = self.p_x.sample((N,))
        logw_denominator = self.standard_normal.log_prob(y.unsqueeze(1) - x.unsqueeze(0))

        return logw_numerator, logw_denominator

    def generate_ys_thetas(self, dataset_size):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, size of the dataset to be generated
        :return: the dataset in a form of a dict.
        """
        theta = self.p_theta.sample((dataset_size,))
        x = self.p_x.sample((dataset_size,))
        y = x + self.standard_normal.sample((dataset_size,))
        return {'y': y, 'theta': theta}

    def p_xy_log_prob(self, x, y):
        return self.p_x.log_prob(x) + self.standard_normal.log_prob(y - x)

    def samples_for_evaluation_helper(self, q1_or_q2, num_of_samples, y_theta):
        y, theta = (y_theta[k] for k in ['y', 'theta'])

        if q1_or_q2 == 'q1':
            conditioned_on = (y, theta)
            q = self.q1
        elif q1_or_q2 == 'q2':
            conditioned_on = (y,)
            q = self.q2
        else:
            raise Exception(f'Unrecognized value of the argument q1_or_q2: {q1_or_q2}.')

        x_samples, q_x_log_prob = q.sample(num_of_samples,
                                           conditioned_on=conditioned_on,
                                           return_logprobs=True)

        f_x_samples = self.f(x_samples, theta)

        p_xy_log_prob = self.p_xy_log_prob(x_samples, y)

        log_w_q = p_xy_log_prob - q_x_log_prob

        x_samples = {'x': x_samples}

        return x_samples, f_x_samples, log_w_q, q_x_log_prob

    def samples_for_evaluation(self, num_of_samples, y_theta, device='cpu'):
        # In 'q_x_log_prob_q1_x1': 'q1' stands for log probs evaluated using distribution q1,
        #  and 'x1' stands for 'samples from distribution q1'
        x_samples_q1, f_x_samples_q1, log_w_q1, q_x_log_prob_q1_x1 = \
            self.samples_for_evaluation_helper('q1', num_of_samples, y_theta)
        x_samples_q2, f_x_samples_q2, log_w_q2, q_x_log_prob_q2_x2 = \
            self.samples_for_evaluation_helper('q2', num_of_samples, y_theta)

        # In 'q_x_log_prob_q1_x2': 'q1' stands for log probs evaluated using distribution q1,
        #  and 'x2' stands for 'samples from distribution q2'
        # These quantities are needed to compute the mixture proposal q_m estimate.
        y, theta = (y_theta[k] for k in ['y', 'theta'])
        q_x_log_prob_q1_x2 = self.q1.log_prob(x_samples_q2['x'], conditioned_on=(y, theta))
        q_x_log_prob_q2_x1 = self.q2.log_prob(x_samples_q1['x'], conditioned_on=(y,))

        output_dict_q1 = {
            'f_x_samples_q1': f_x_samples_q1,
            'log_w_q1': log_w_q1,
            'q_x_log_prob_q1_x1': q_x_log_prob_q1_x1,
            'q_x_log_prob_q1_x2': q_x_log_prob_q1_x2,
        }

        output_dict_q2 = {
            'f_x_samples_q2': f_x_samples_q2,
            'log_w_q2': log_w_q2,
            'q_x_log_prob_q2_x1': q_x_log_prob_q2_x1,
            'q_x_log_prob_q2_x2': q_x_log_prob_q2_x2,
        }

        output_dict_q1, output_dict_q2 = map(lambda x: {k: v.to(device) for k, v in x.items()},
                                             (output_dict_q1, output_dict_q2))

        return output_dict_q1, output_dict_q2

    def load_ys_thetas_and_groundtruths(self, dataset_size):
        if self.d == 1:
            ys_thetas = self.generate_ys_thetas(dataset_size)
            ys, thetas = (ys_thetas[k] for k in ('y', 'theta'))
            # This variant with pytorch doesn't provide sufficient numerical precision
            #  ground_truths = 1. - torch.distributions.Normal(ys / 2., 1. / np.sqrt(2.)).cdf(thetas)
            # The line below is coming from the analytical solution.
            ground_truths = 1. - scipy.stats.norm.cdf(thetas.to('cpu').numpy().astype(np.float64),
                                                      loc=ys.to('cpu').numpy().astype(np.float64)/2.,
                                                      scale=1./np.sqrt(2.))
            if (ground_truths == 0.).any():
                raise Exception("Ground truth should never be 0, numerical issues.")
            return ys_thetas, ground_truths
        elif self.d == 5:
            return amci.utils.load_ys_thetas_and_groundtruths(self, dataset_size)
