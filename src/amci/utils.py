import collections
import datetime
import importlib
import math
import os.path
import pprint
import random
import subprocess
import sys

import git
import numpy as np
import sacred
import tensorboardX
import torch
import tqdm


def git_revision():
    return subprocess.check_output("git rev-parse --short HEAD".split()).strip().decode("utf-8")


def cat_last_dim(x, with_expand=False):
    """
    Concatenates a sequence of tensors along the last dimension, if with_expand==True it expands/broadcasts the tensors
    as necessary for concatenation.
    :param x: sequence of tensors of the same number of dimensions.
    :param with_expand: boolean deciding whether to expand/broadcast the tensors as necessary for concatenation.
    :return: concatenated tensor
    """
    assert all(len(i.shape) == 2 for i in x)

    if not with_expand:
        return torch.cat(x, dim=-1)

    shape_to_expand = np.array(tuple(tuple(i.shape[:-1]) for i in x)).max(axis=0).tolist()

    return cat_last_dim(tuple(i.expand(*shape_to_expand, -1) for i in x), with_expand=False)


def flatten(l):
    """
    Flattens the list l.
    :param l: list
    :return: flattened list.
    """
    return [item for sublist in l for item in sublist]


def iterate_minibatches(batchsize, data):
    assert len(data) > 0
    assert isinstance(data, collections.abc.Mapping)

    for start_idx in range(0, len(data[tuple(data.keys())[0]]) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield {k: v[excerpt] for k, v in data.items()}


def experiment_name(args):
    """
    Returns the experiment name.
    """
    if args.problem_name in ['tail_integral_1d', 'tail_integral_5d']:
        return f'{args.problem_name}_{args.q1_or_q2}'
    elif args.problem_name == 'cancer':
        return f'{args.problem_name}_{args.factor}_{args.q1_or_q2}'
    else:
        raise Exception(f'Unknown problem_name: {args.problem_name}.')


def get_model(args):
    """
    Imports appropriate module for the model.
    """
    if args.problem_name in ['tail_integral_1d', 'tail_integral_5d']:
        return importlib.import_module(f'amci.tail_integral.model')
    elif args.problem_name == 'cancer':
        return importlib.import_module(f'amci.cancer.model')
    else:
        raise Exception(f'Unknown problem_name: {args.problem_name}.')


def sacred_main_helper(train_func, args, _run):
    """
    Helper function
    """
    for k, v in args.__dict__.items():
        if type(v) == sacred.config.custom_containers.ReadOnlyList:
            setattr(args, k, list(v))

    args.experiment_name = experiment_name(args)

    # Handle git repo related aspects
    repo = git.Repo('../..')
    args.git_clean = not repo.is_dirty()
    if not args.git_clean and args.require_clean_repo:
        raise RuntimeError("The repo is not clean, change require_clean_repo flag if you want to"
                           "run the code with a dirty repo.")
    args.git_commit = git_revision()

    # Handle CUDA config
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor')
    args.device = 'cuda' if args.cuda else 'cpu'

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Sort out paths
    args.runid = datetime.datetime.now().strftime("%y%m%d_%H%M_%f") + '_' + args.git_commit
    if args.logs_root is None:
        # os.path.dirname(sys.argv[0]) should point to amci/src/amci
        args.logs_root = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '../../logs'))
    args.output_folder = os.path.join(args.logs_root, f'{args.experiment_name}_{args.runid}')
    os.mkdir(args.output_folder)
    _run.info['output_folder'] = os.path.abspath(args.output_folder)
    _run.info['args'] = args.__dict__

    # Tensorboard
    _writer = tensorboardX.SummaryWriter(os.path.join(args.output_folder, 'tensorboard.file'))
    _writer.add_text('hyperparameters', pprint.pformat(args.__dict__))

    with open(os.path.join(args.output_folder, 'config.txt'), 'wt') as out:
        pprint.pprint(args.__dict__, stream=out)

    pprint.pprint(args.__dict__)

    return train_func(args, _run, _writer)


def validate_checkpoint(checkpoint_filepath):
    if not (os.path.isfile(checkpoint_filepath) and os.access(checkpoint_filepath, os.R_OK)):
        raise Exception(f'Cannot access the checkpoint, it doesnt exist or dont have the right permissions: '
                        f'{checkpoint_filepath}.')


def validate_checkpoints(args):
    validate_checkpoint(args.checkpoint_q1)
    validate_checkpoint(args.checkpoint_q2)


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    any_key_of_the_dict = list(dict_of_lists.keys())[0]

    list_of_dicts = [{k: v[i:i + 1] for k, v in dict_of_lists.items()}
                     for i in range(len(dict_of_lists[any_key_of_the_dict]))]

    return list_of_dicts


def cuda_config(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor')
    return args


def generate_samples_for_evaluation(number_of_samples_total, number_of_samples_gpu_capacity, ys_thetas, model,
                                    device='cpu'):
    data_dicts_q1, data_dicts_q2 = [], []

    for y_theta in tqdm.tqdm(dict_of_lists_to_list_of_dicts(ys_thetas)):
        data_dicts_q1_temp = []
        data_dicts_q2_temp = []

        number_of_samples_left = number_of_samples_total

        for _ in tqdm.tqdm(range(int(math.ceil(number_of_samples_total/number_of_samples_gpu_capacity))),
                           disable=False):
            # returns a dict
            data_dict_q1, data_dict_q2 = model.samples_for_evaluation(
                min(number_of_samples_left, number_of_samples_gpu_capacity),
                y_theta=y_theta, device=device)

            data_dicts_q1_temp.append(data_dict_q1)
            data_dicts_q2_temp.append(data_dict_q2)

            number_of_samples_left -= number_of_samples_gpu_capacity

        data_dicts_q1.append({k: torch.cat(tuple(d[k] for d in data_dicts_q1_temp), dim=0) for k in data_dicts_q1_temp[0].keys()})
        data_dicts_q2.append({k: torch.cat(tuple(d[k] for d in data_dicts_q2_temp), dim=0) for k in data_dicts_q2_temp[0].keys()})

    # lists of dicts
    data_dict_q1, data_dict_q2 = map(
        lambda data_dicts: {k: torch.stack(tuple(d[k] for d in data_dicts), dim=0) for k in data_dicts[0].keys()},
        (data_dicts_q1, data_dicts_q2))

    return data_dict_q1, data_dict_q2


def get_flow_hyper_net(hidden_units_per_layer, parameters_nelement, in_dim, number_of_layers=2):
    """
    Returns an MLP neural net used as the hyper-network for the flows.
    """
    H = hidden_units_per_layer

    layers = [torch.nn.Linear(in_dim, H), torch.nn.ReLU()]
    layers += flatten([[torch.nn.Linear(H, H), torch.nn.ReLU()] for _ in range(number_of_layers)])
    layers += [torch.nn.Linear(H, parameters_nelement),]

    flow_hyper_net = torch.nn.Sequential(*layers)

    for param in flow_hyper_net.parameters():
        torch.nn.init.uniform_(param, a=-0.01, b=0.01)

    # Maybe in more complicated cases the hypernetwork could benefit from a more sophisticated initialization strategy,
    #  e.g. like or inspired by Chang et al., Principled Weight Initialization for Hypernetworks, ICLR 2020
    #  https://openreview.net/forum?id=H1lma24tPB
    # It could also use normalizing the input to mean 0, std 1, e.g. by utilizing the knowledge about the prior
    #  distributions over the variables we're conditioning over. That would partially alleviate the need for such
    #  "hand-crafted" (chosen the magnitude heuristically) initialization like here.

    return flow_hyper_net


def load_proposals_from_checkpoints(object, factors, model_parameters=tuple()):
    """
    Loads checkpoints for the proposals.
    Checks the consistency of the model parameters between different checkpoints loaded.
    Note: Training and Evaluation should inherit from some common abstract base class.
    :param object: instantiation of the Evaluation class
    :param factors: collection of factor names to be loaded
    :param model_parameters: collection of model parameter names
    :return: model parameters consistent for all the checkpoints
    """
    checkpoints_args_dicts = []
    for factor in factors:
        checkpoint = torch.load(object.args.__dict__[f'checkpoint_{factor}'], map_location=object.args.device)
        object.__setattr__(factor, object.get_proposal_model(checkpoint[1]))
        object.__getattribute__(factor).load_state_dict(checkpoint[0])
        checkpoints_args_dicts.append(checkpoint[-2].__dict__)
    # checks the consistency of the model parameters between different checkpoints loaded
    for k in model_parameters:
        v = checkpoints_args_dicts[0][k]
        for checkpoints_args_dict in checkpoints_args_dicts[1:]:
            assert checkpoints_args_dict[k] == v
    return {k: checkpoints_args_dicts[0][k] for k in model_parameters}


def load_ys_thetas_and_groundtruths(self, dataset_size):
    """
    Loads ys_thetas and ground truth values from the file as per run configuration.
    Note: Training and Evaluation should inherit from some common abstract base class.
    """
    filepath = os.path.join(self.args.logs_root,
                            f'ground_truths_{self.args.problem_name}_{str(self.args.ground_truth_samples)}')
    if not (os.path.isfile(filepath) and os.access(filepath, os.R_OK)):
        raise Exception(f'Cannot access the groundtruths file {filepath}, '
                        f'it doesnt exist or you dont have the right permissions. '
                        f'You can generate the groundtruths file using ground_truth.py script.')

    ground_truths_dict = torch.load(filepath)
    print(f'Loading ground truths from :{filepath}')

    # check that the ground truth was generated for the same model parameter values as for
    for k, v in self.loaded_checkpoint_parameters.items():
        assert ground_truths_dict['args'].__dict__[k] == v

    if tuple(ground_truths_dict['ys_thetas'].values())[0].shape[0] != dataset_size:
        raise Exception(f'The number of datapoints in file {filepath} ({ground_truths_dict["ys_thetas"].shape[0]})'
                        f'does not match the argument dataset_size ({dataset_size}).')

    return ground_truths_dict['ys_thetas'], ground_truths_dict['estimate']
