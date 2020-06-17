import os.path
import pprint
from collections import defaultdict
from types import SimpleNamespace

import configargparse
import hdf5storage
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import yaml

import amci.utils

matplotlib.rc('text', usetex=True)
plt.rc('font', family='serif')


def main():
    parser = configargparse.get_argument_parser()

    parser.add('config')

    with open(parser.parse_known_args()[0].config, 'r') as stream:
        args = SimpleNamespace(**yaml.safe_load(stream)['config'])

    args.no_cuda = True
    args = amci.utils.cuda_config(args)
    torch.set_grad_enabled(False)

    args.figure_remse_number_of_tries = int(float(args.figure_remse_number_of_tries))
    args.figure_remse_number_of_y_theta_samples = int(float(args.figure_remse_number_of_y_theta_samples))
    args.figure_remse_points_to_be_displayed_with_a_log_scale = int(float(args.figure_remse_points_to_be_displayed_with_a_log_scale))
    args.figure_remse_xaxis_max_samples = int(float(args.figure_remse_xaxis_max_samples))

    pprint.pprint(args.__dict__)

    plot(args)


def plot(args):
    variable_names = [
        'f_x_samples_q1',
        'log_w_q1',
        'q_x_log_prob_q1_x1',
        'q_x_log_prob_q1_x2',
        'f_x_samples_q2',
        'log_w_q2',
        'q_x_log_prob_q2_x1',
        'q_x_log_prob_q2_x2',
        'ground_truths',
    ]

    # maximum number of samples used for a single estimator (maximum value on the x-axis in the plot)
    #  it is the total number of samples generated divided by the number of trials
    N = int(args.figure_remse_xaxis_max_samples)

    data = {}
    data.update(hdf5storage.loadmat(os.path.join(os.path.dirname(args.checkpoint_q1), 'samples.mat'),
                                    variable_names=variable_names[:4]))
    data.update(hdf5storage.loadmat(os.path.join(os.path.dirname(args.checkpoint_q2), 'samples.mat'),
                                    variable_names=variable_names[4:]))

    if (data['ground_truths'] == 0.).any():
        raise Exception("Ground truth should never be 0, numerical isues.")

    for k, v in data.items():
        v[np.isnan(v)] = float('-inf')

    data = {k: v.astype(np.float64) for k, v in data.items()}

    ground_truths = data['ground_truths'].reshape(-1, 1)

    # Some flows happen to might yield nan when the log_prob is -inf
    for k, v in data.items():
        v[np.isnan(v)] = float('-inf')

    # Every key of data dictionary is of shape [figure_remse_number_of_y_theta_samples, N * tries]
    o = SimpleNamespace()
    _ = [o.__setattr__(k, v) for k, v in data.items() if k in variable_names[:8]]

    # In the name 'q_x_log_prob_q1_x2': 'q1' stands for log probs evaluated using distribution q1,
    #  and 'x2' stands for 'samples from distribution q2'.

    # Estimate SNIS lower bound per Equation 5 using importance sampling, assuming proposal q_m = 0.5*q_1 + 0.5*q_2
    #  since half of the samples come from q1, half from q2, and we want to make use of all the samples.
    log_p = np.concatenate([o.log_w_q1 + o.q_x_log_prob_q1_x1,
                            o.log_w_q2 + o.q_x_log_prob_q2_x2],
                           axis=1)
    log_q = np.concatenate((scipy.special.logsumexp(np.stack((o.q_x_log_prob_q1_x1, o.q_x_log_prob_q2_x1), axis=0), axis=0),
                            scipy.special.logsumexp(np.stack((o.q_x_log_prob_q1_x2, o.q_x_log_prob_q2_x2), axis=0), axis=0)),
                           axis=1) + np.log(0.5)
    ws = np.exp(log_p - log_q)
    w_bars = ws / np.sum(ws, axis=1, keepdims=True)
    fs = np.concatenate([o.f_x_samples_q1, o.f_x_samples_q2], axis=1)
    expectation_abs_errors = np.sum(w_bars * np.abs(fs - ground_truths), axis=1, keepdims=True)

    # Bound from Equation 5
    mse_snis_bound = expectation_abs_errors ** 2 / np.arange(1, N + 1).reshape(1, -1)

    # Relative to ground truths
    remse_snis_bound = mse_snis_bound / (ground_truths ** 2)

    t = SimpleNamespace()

    # reshape
    [t.__setattr__(k, v.reshape((args.figure_remse_number_of_y_theta_samples, args.figure_remse_number_of_tries, -1)))
     for k, v in o.__dict__.items()]

    # Compute the Mixture IS with the proposal qm = 0.5*q1 + 0.5*q2
    log_p_x1 = t.log_w_q1 + t.q_x_log_prob_q1_x1
    log_p_x2 = t.log_w_q2 + t.q_x_log_prob_q2_x2

    # _j stands for joint - it is a combination of samples fromm q1 and q2
    log_p_j = np.concatenate([log_p_x1, log_p_x2], axis=2)
    f_j = np.concatenate([t.f_x_samples_q1, t.f_x_samples_q2], axis=2)
    log_q1_j = np.concatenate([t.q_x_log_prob_q1_x1, t.q_x_log_prob_q1_x2], axis=2)
    log_q2_j = np.concatenate([t.q_x_log_prob_q2_x1, t.q_x_log_prob_q2_x2], axis=2)

    # shape: [figure_remse_number_of_y_theta_samples, tries, 2*N, 4]
    to_be_shuffled = np.stack([log_p_j, f_j, log_q1_j, log_q2_j], axis=3)

    shp = to_be_shuffled.shape[:-2]
    for ndx in np.ndindex(shp):
        np.random.shuffle(to_be_shuffled[ndx])

    # From here '_qm' denotes regarding the mixture proposal qm = 0.5*q1 + 0.5*q2
    log_p_qm = to_be_shuffled[:, :, :N, 0]
    f_qm = to_be_shuffled[:, :, :N, 1]

    log_q1_j = to_be_shuffled[:, :, :N, 2] + np.log(0.5)
    log_q2_j = to_be_shuffled[:, :, :N, 3] + np.log(0.5)

    log_w_qm = log_p_qm - scipy.special.logsumexp(np.stack([log_q1_j, log_q2_j], axis=3), axis=3)

    # Relative error in the numerator estimate
    temp = (('q1', t.f_x_samples_q1, t.log_w_q1),
            ('q2', t.f_x_samples_q2, t.log_w_q2),
            ('qm', f_qm, log_w_qm))

    # use torch.logcumsumexp here when it becomes operational
    #  https://github.com/pytorch/pytorch/pull/32876
    running_estimate_n = {k: np.cumsum(np.exp(log_w) * f_x, axis=2) for k, f_x, log_w in temp}
    running_estimate_d = {k: np.cumsum(np.exp(log_w), axis=2) for k, _, log_w in temp}

    temp = (('amci', 'q1', 'q2'),
            ('snis', 'q2', 'q2'),
            ('snis_q1', 'q1', 'q1'),
            ('mixture', 'qm', 'qm'))

    running_estimate = {k: running_estimate_n[k_n] / running_estimate_d[k_d] for k, k_n, k_d in temp}

    remse = {k: (np.abs(v - ground_truths.reshape(-1, 1, 1))/ground_truths.reshape(-1, 1, 1)) ** 2
             for k, v in running_estimate.items()}

    # Thinning the data series to be plotted with a log scale
    xs = np.interp(
        np.linspace(np.log(1.), np.log(N), args.figure_remse_points_to_be_displayed_with_a_log_scale),
        np.log(np.arange(1, N + 1)),
        np.arange(1, N + 1))
    xs = np.unique(np.round(xs)).astype(np.int) - 1

    remse['bound'] = remse_snis_bound.reshape(args.figure_remse_number_of_y_theta_samples, 1, -1)

    remse = {k: remse[:, :, xs] for k, remse in remse.items()}

    # Plot style
    colors = {'AMCI': 'b',
              'SNIS $q_2$': 'r',
              'SNIS $q_m$': 'g',
              'SNIS $q_1$': 'm',
              'SNIS bound': 'k',}
    linestyles = defaultdict(lambda: None)
    linestyles['SNIS bound'] = '--'
    alpha = 0.2
    shaded_quantile_size = 0.25

    label_size = 26
    ticks_size = 22
    legend_size = 20

    reduce_datapoints = np.median
    reduce_trials = np.mean
    plot_bands = True

    plot_args = (reduce_datapoints, reduce_trials, plot_bands)

    def plot_helper_final(thinned_error, label, ax, reduce_datapoints=np.mean, reduce_trials=np.mean, plot_bands=True):
        xs_multiplier = 1
        if label in ['AMCI']:
            xs_multiplier = 2

        yt = reduce_trials(thinned_error, axis=1)
        y = reduce_datapoints(yt, axis=0)
        quantiles_over_trials = np.quantile(thinned_error,
                                            [shaded_quantile_size,
                                             1. - shaded_quantile_size],
                                            axis=1)
        u = reduce_datapoints(quantiles_over_trials[0], axis=0)
        l = reduce_datapoints(quantiles_over_trials[1], axis=0)

        outputs = []
        outputs.append(ax.loglog(xs_multiplier * (xs + 1), y,
                                 label=label,
                                 color=colors[label],
                                 linestyle=linestyles[label]))
        if plot_bands:
            outputs.append(ax.fill_between(xs_multiplier * (xs + 1), u, l,
                                           color=colors[label], alpha=alpha))
        return outputs

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    axt = ax
    plot_helper_final(remse['amci'], 'AMCI', axt, *plot_args)
    plot_helper_final(remse['snis'], 'SNIS $q_2$', axt, *plot_args)
    plot_helper_final(remse['mixture'], 'SNIS $q_m$', axt, *plot_args)
    plot_helper_final(remse['bound'], 'SNIS bound', axt, *plot_args)

    axt.set_xlim((2, 1e4))
    axt.set_ylim(float(args.figure_remse_ylim_lower), float(args.figure_remse_ylim_upper))

    axt.tick_params(labelsize=ticks_size)

    axt.set_xlabel('Number of samples $N$', size=label_size)
    axt.set_ylabel('ReMSE', size=label_size)

    axt.tick_params(labelsize=ticks_size)

    _ = axt.legend(loc=3, fontsize=legend_size, handlelength=0.9)

    fig.tight_layout()

    fig.savefig(os.path.join(os.path.dirname(args.checkpoint_q1), 'figure_remse.pdf'))
    fig.savefig(os.path.join(os.path.dirname(args.checkpoint_q2), 'figure_remse.pdf'))

    msg = f'checkpoint_q1: {args.checkpoint_q1}\ncheckpoint_q2: {args.checkpoint_q2}'
    with open(os.path.join(os.path.dirname(args.checkpoint_q1), 'figure_checkpoints.txt'), 'w') as file:
        file.write(msg)
    with open(os.path.join(os.path.dirname(args.checkpoint_q2), 'figure_checkpoints.txt'), 'w') as file:
        file.write(msg)

    print(f'Saved at {os.path.join(os.path.dirname(args.checkpoint_q1), "figure_remse.pdf")}')
    print(f'Saved at {os.path.join(os.path.dirname(args.checkpoint_q2), "figure_remse.pdf")}')


if __name__ == '__main__':
    main()
