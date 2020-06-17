import torch
import torchdiffeq
from torch import tensor

import amci.flows
import amci.utils


class Training:
    def __init__(self, q1_or_q2, q, args):
        if q1_or_q2 not in ['q1', 'q2']:
            raise Exception(f'Unrecognized value of the argument q1_or_q2: {q1_or_q2}.')

        self.q1_or_q2 = q1_or_q2

        self.q = q

        self.args = args

        self.init_distributions()

    def init_distributions(self):
        args = self.args

        # torch.distributions.independent.Independent allows to set event_shape=(1,) for the distribution
        self.p_eps = torch.distributions.independent.Independent(
            torch.distributions.Beta(args.eps_a * torch.ones((1,)), args.eps_b * torch.ones((1,))), 1)
        self.p_c0 = torch.distributions.independent.Independent(
            torch.distributions.Gamma(args.c0k * torch.ones((1,)), 1. / args.c0theta * torch.ones((1,))), 1)

        self.c_obs_dist = lambda c: torch.distributions.independent.Independent(
            torch.distributions.Gamma(c ** 2 / args.c_obs_sd ** 2, c / args.c_obs_sd ** 2), 1)

    @staticmethod
    def get_proposal_model(args):
        """
        Returns the proposal distribution q.
        """

        # specification of the flow
        if args.factor == 'eps':
            hyper_net_in_dim = 3
            parameters_nelement = 2
            density_esimator = amci.flows.Beta()
        elif args.factor == 'c0':
            hyper_net_in_dim = 2
            parameters_nelement = 2
            density_esimator = amci.flows.Gamma()
        else:
            raise Exception(f'Unrecognized value of the argument factor: {args.factor}.')

        hyper_net = amci.utils.get_flow_hyper_net(args.hidden_units_per_layer,
                                                  parameters_nelement,
                                                  hyper_net_in_dim,
                                                  number_of_layers=args.neural_net_depth)
        q = amci.flows.ConditionedDensityEstimator(hyper_net=hyper_net, density_estimator=density_esimator)

        return q

    @staticmethod
    def tumor_ode(y, eps):
        phi = 5.85
        psi = 0.00873
        lambd = 0.1923

        # Passing c and K toghether as part of a single parameter y is necessary due to torchdiffeq.odeint API
        c, K = y.split(1, dim=1)
        return torch.cat(
            (- lambd * c * (c / K).log() - eps * c,
             phi * c - psi * K * c ** (2. / 3.)),
            dim=1)

    @staticmethod
    def tumor_evolution(c, k, eps, t):
        t = tensor([0., t])
        y0 = torch.cat([c, k], dim=1)
        c, k = torchdiffeq.odeint(lambda t, y: Training.tumor_ode(y=y, eps=eps), y0, t)[-1].split(1, dim=1)

        return c, k

    def f(self, c):
        floor = float(self.args.function_floor)

        surgery_success_probability = (torch.tanh(-(c - self.args.tanh_centre) / self.args.tanh_scale)
                                       + 1.) / (2. / (1. - 2. * floor)) + floor

        return surgery_success_probability.reshape(-1)

    def generate_dataset(self, dataset_size, with_f_x=None):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, number of samples to be generated
        :param with_f_x: whether to also return value of f_x. At times we might want not to if it's expensive.
        :return: the dataset in a form of a dict.
        """
        if with_f_x is None:
            with_f_x = (self.q1_or_q2 == 'q1')

        eps = self.p_eps.sample((dataset_size,))
        c0 = self.p_c0.sample((dataset_size,))
        k0 = 700. * torch.ones_like(c0)

        c1, k1 = self.tumor_evolution(c=c0, eps=eps, k=k0, t=self.args.tcheckup)

        c0o = self.c_obs_dist(c0).sample()
        c1o = self.c_obs_dist(c1).sample()

        if with_f_x:
            # final size of the tumor
            cf, _ = self.tumor_evolution(c=c1, eps=eps, k=k1,
                                         t=self.args.tmax-self.args.tcheckup)
            f_x = self.f(cf)
            return {'eps': eps, 'c0': c0, 'c1': c1, 'c0o': c0o, 'c1o': c1o, 'f_x': f_x}
        else:
            return {'eps': eps, 'c0': c0, 'c1': c1, 'c0o': c0o, 'c1o': c1o}

    def loss(self, c0o=None, c1o=None, c0=None, eps=None, f_x=None, **kwargs):
        """
        Computes the loss function.
        """
        if self.args.factor == 'c0':
            conditioned_on = (c0o, c1o)
            x = c0
        elif self.args.factor == 'eps':
            conditioned_on = (c0o, c1o, c0)
            x = eps
        else:
            raise Exception(f'Unknown factor: {self.args.factor}')

        if self.q1_or_q2 == 'q1':
            loss = -f_x * self.q.log_prob(x, conditioned_on=conditioned_on)
        elif self.q1_or_q2 == 'q2':
            loss = -self.q.log_prob(x, conditioned_on=conditioned_on)

        return loss


class Evaluation(Training):
    def __init__(self, args, load_checkpoints=True):
        self.args = args

        self.init_distributions()

        if load_checkpoints:
            model_parameters_names = ('eps_a', 'eps_b', 'c0k', 'c0theta', 'c_obs_sd')
            self.loaded_checkpoint_parameters = \
                amci.utils.load_proposals_from_checkpoints(self, ('q1_c0', 'q1_eps', 'q2_c0', 'q2_eps'),
                                                           model_parameters_names)

    def logw_for_groundtruth(self, N, y_theta):
        data = self.generate_dataset(N, with_f_x=True)

        # the shape of logw_denominator is [figure_remse_number_of_y_theta_samples, ground_truth_capacity]
        logw_denominator = self.c_obs_dist(data['c0'].unsqueeze(0)).log_prob(y_theta['c0o'].unsqueeze(1)) +\
                           self.c_obs_dist(data['c1'].unsqueeze(0)).log_prob(y_theta['c1o'].unsqueeze(1))
        logw_numerator = logw_denominator + data['f_x'].log().unsqueeze(0)

        return logw_numerator, logw_denominator

    def generate_ys_thetas(self, dataset_size):
        """
        Generates a dataset of a given size.
        :param dataset_size: int, size of the dataset to be generated
        :return: the dataset in a form of a dict.
        """
        data = self.generate_dataset(dataset_size, with_f_x=False)

        return {k: data[k] for k in ['c0o', 'c1o']}

    def p_xy_log_prob(self, eps, c0, c1, c0o, c1o):
        return self.p_eps.log_prob(eps) + self.p_c0.log_prob(c0) +\
               self.c_obs_dist(c0).log_prob(c0o) + self.c_obs_dist(c1).log_prob(c1o)

    def samples_for_evaluation_helper(self, q1_or_q2, num_of_samples, y_theta):
        c0o, c1o = (y_theta[k] for k in ['c0o', 'c1o'])

        if q1_or_q2 == 'q1':
            q_c0 = self.q1_c0
            q_eps = self.q1_eps
        elif q1_or_q2 == 'q2':
            q_c0 = self.q2_c0
            q_eps = self.q2_eps
        else:
            raise Exception(f'Unrecognized value of the argument q1_or_q2: {q1_or_q2}.')

        # c0o, c1o is of shape [1, dimensionality of variable]
        # c0 should be of shape [num_of_samples, dimensionality of c0]
        c0 = q_c0.sample(num_of_samples, conditioned_on=(c0o, c1o))
        eps = q_eps.sample(conditioned_on=(c0o, c1o, c0))
        k0 = 700. * torch.ones_like(c0)

        x_samples = {'c0': c0, 'eps': eps}

        c1, k1 = self.tumor_evolution(c=c0, eps=eps, k=k0, t=self.args.tcheckup)

        # final size of the tumor
        cf, _ = self.tumor_evolution(c=c1, eps=eps, k=k1, t=self.args.tmax - self.args.tcheckup)
        f_x_samples = self.f(cf)

        p_xy_log_prob = self.p_xy_log_prob(eps, c0, c1, c0o, c1o)

        q_x_log_prob = q_c0.log_prob(c0, conditioned_on=(c0o, c1o)) + \
                       q_eps.log_prob(eps, conditioned_on=(c0o, c1o, c0))

        log_w_q = p_xy_log_prob - q_x_log_prob

        return x_samples, f_x_samples, log_w_q, q_x_log_prob

    def samples_for_evaluation(self, num_of_samples, y_theta, device='cpu'):
        c0o, c1o = (y_theta[k] for k in ['c0o', 'c1o'])

        # In 'q_x_log_prob_q1_x1': 'q1' stands for log probs evaluated using distribution q1,
        #  and 'x1' stands for 'samples from distribution q1'
        x_samples_q1, f_x_samples_q1, log_w_q1, q_x_log_prob_q1_x1 = \
            self.samples_for_evaluation_helper('q1', num_of_samples, y_theta)
        x_samples_q2, f_x_samples_q2, log_w_q2, q_x_log_prob_q2_x2 = \
            self.samples_for_evaluation_helper('q2', num_of_samples, y_theta)

        # In 'q_x_log_prob_q1_x2': 'q1' stands for log probs evaluated using distribution q1,
        #  and 'x2' stands for 'samples from distribution q2'
        # These quantities are needed to compute the mixture proposal q_m estimate.
        q_x_log_prob_q1_x2 = self.q1_c0.log_prob(x_samples_q2['c0'], conditioned_on=(c0o, c1o)) + \
                             self.q1_eps.log_prob(x_samples_q2['eps'], conditioned_on=(c0o, c1o, x_samples_q2['c0']))
        q_x_log_prob_q2_x1 = self.q2_c0.log_prob(x_samples_q1['c0'], conditioned_on=(c0o, c1o)) + \
                             self.q2_eps.log_prob(x_samples_q1['eps'], conditioned_on=(c0o, c1o, x_samples_q1['c0']))

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
        return amci.utils.load_ys_thetas_and_groundtruths(self, dataset_size)
