import math
import os.path
import pprint
import sys
from types import SimpleNamespace

import configargparse
import torch
import tqdm
import yaml

import amci.utils


def main():
    parser = configargparse.get_argument_parser()

    parser.add('config')

    with open(parser.parse_known_args()[0].config, 'r') as stream:
        args = SimpleNamespace(**yaml.safe_load(stream)['config'])

    args = amci.utils.cuda_config(args)
    torch.set_grad_enabled(False)

    if args.logs_root is None:
        args.logs_root = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '../../logs'))
    args.filepath = os.path.join(args.logs_root,
                                 f'ground_truths_{args.problem_name}_{str(args.ground_truth_samples)}')

    args.ground_truth_samples = int(float(args.ground_truth_samples))
    args.ground_truth_capacity = int(float(args.ground_truth_capacity))

    pprint.pprint(args.__dict__)

    ground_truth(args)


def ground_truth(args):
    model = amci.utils.get_model(args)

    evaluation = model.Evaluation(args, load_checkpoints=False)
    ys_thetas = evaluation.generate_ys_thetas(args.figure_remse_number_of_y_theta_samples)

    iterations = int(math.ceil(args.ground_truth_samples / args.ground_truth_capacity))
    samples_left = args.ground_truth_samples

    temp_numerator = []
    temp_denominator = []
    temp_2times_denominator = []
    for _ in tqdm.tqdm(range(iterations)):
        N = min(samples_left, args.ground_truth_capacity)
        logw_numerator, logw_denominator = evaluation.logw_for_groundtruth(N, ys_thetas)
        samples_left -= args.ground_truth_capacity

        convert_to = torch.float64
        temp_numerator.append(torch.logsumexp(logw_numerator.to(convert_to), dim=1, keepdim=True))
        temp_denominator.append(torch.logsumexp(logw_denominator.to(convert_to), dim=1, keepdim=True))
        temp_2times_denominator.append(torch.logsumexp(2. * logw_denominator.to(convert_to), dim=1, keepdim=True))

    logsumexp_log_numerator = torch.logsumexp(torch.cat(temp_numerator, dim=1).to(torch.float64), dim=1)
    logsumexp_log_denominator = torch.logsumexp(torch.cat(temp_denominator, dim=1).to(torch.float64), dim=1)
    estimate_numerator = (logsumexp_log_numerator - math.log(args.ground_truth_samples)).exp()
    estimate_denominator = (logsumexp_log_denominator - math.log(args.ground_truth_samples)).exp()
    estimate = (logsumexp_log_numerator - logsumexp_log_denominator).exp()

    # Estimated samples size
    ess = (2. * logsumexp_log_denominator -
           torch.logsumexp(torch.cat(temp_2times_denominator, dim=1), dim=1)).exp()
    print(f'Estimated samples size: {ess}')

    output = dict()
    output['ys_thetas'] = ys_thetas
    output['estimate_numerator'] = estimate_numerator
    output['estimate_denominator'] = estimate_denominator
    output['estimate'] = estimate
    output['args'] = args
    output['ESS'] = ess

    print(f'Saving results to file: {args.filepath}')
    torch.save(output, args.filepath)

    return estimate


if __name__ == '__main__':
    main()
