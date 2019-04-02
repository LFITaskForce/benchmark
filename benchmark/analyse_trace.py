import torch
from torch.autograd import grad
import inspect


def calculate_joint_score(trace, params):
    score = 0.
    for dist, z, _ in _get_branchings(trace):
        score = score + _individual_score(dist, z, params)
    return score


def _get_branchings(trace):
    for key in trace.nodes:
        if key in ["_INPUT", "_RETURN"]:
            continue
        node = trace.nodes[key]

        dist = node["fn"]
        z = node["value"]

        param_names = _get_param_names(dist)
        params = []
        for param_name in _get_param_names(dist):
            param = getattr(dist, param_name)
            if len(param.size()) == 1:
                param = param.view(-1, 1)
            params.append(param)
        params = torch.cat(params, 1)

        yield dist, z, params


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


def _individual_log_ratio(dist, z, theta0, theta1):
    if theta0 is None:
        log_p0 = dist.log_prob(z)
    else:
        pass
        # TODO
        # look up replay() in Pyro Poutine

    # TODO

    return log_p0 - log_p1


def _get_param_names(distribution):
    param_names_unsorted = list(distribution.arg_constraints.keys())
    sig = inspect.signature(distribution.__init__)

    param_names_sorted = []
    for param in sig.parameters:
        if param in param_names_unsorted:
            param_names_sorted.append(param)
            param_names_unsorted.remove(param)

    return param_names_sorted + param_names_unsorted
