import torch
from torch.autograd import grad
from pyro import poutine
import inspect


def calculate_x(trace):
    node = trace.nodes["_RETURN"]
    x = node["value"]
    return x


def calculate_joint_log_prob(trace):
    log_p = 0.
    for dist, z, _ in _get_branchings(trace):
        log_p = log_p + _individual_log_p(dist, z)
    return log_p


def calculate_joint_score(trace, params):
    score = 0.
    for dist, z, _ in _get_branchings(trace):
        score = score + _individual_score(dist, z, params)
    return score


def calculate_joint_likelihood_ratio(trace, params_num, params_den):
    trace_num = _replay_trace(trace, params_num)
    trace_den = _replay_trace(trace, params_den)

    log_p_num = calculate_joint_log_prob(trace_num)
    log_p_den = calculate_joint_log_prob(trace_den)

    return log_p_num - log_p_den


def _get_branchings(trace):
    for key in trace.nodes:
        if key in ["_INPUT", "_RETURN"]:
            continue
        node = trace.nodes[key]

        dist = node["fn"]
        z = node["value"]

        params = []
        for param_name in _get_param_names(dist):
            param = getattr(dist, param_name)
            if len(param.size()) == 1:
                param = param.view(-1, 1)
            params.append(param)
        params = torch.cat(params, 1)

        yield dist, z, params


def _replay_trace(trace, params):
    if params is None:
        return trace

    # TODO
    replayed_trace = poutine.trace(
        poutine.replay(model.forward, trace)
    )


def _individual_log_p(dist, z):
    return dist.log_prob(z)


def _individual_score(dist, z, theta):
    log_p = dist.log_prob(z)
    score, = grad(
        log_p,
        theta,
        grad_outputs=torch.ones_like(log_p.data),
        only_inputs=True,
        create_graph=False,
    )
    return score


def _get_param_names(distribution):
    param_names_unsorted = list(distribution.arg_constraints.keys())
    sig = inspect.signature(distribution.__init__)

    param_names_sorted = []
    for param in sig.parameters:
        if param in param_names_unsorted:
            param_names_sorted.append(param)
            param_names_unsorted.remove(param)

    return param_names_sorted + param_names_unsorted
