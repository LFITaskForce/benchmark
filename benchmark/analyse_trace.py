import torch
import inspect


def calculate_joint_log_ratio(trace, params0=None, params1=None):
    log_r = 0.
    for dist, z, params in _get_branchings(trace):
        print(dist, z, params)

        #params0_ = _wrap_params(params, params0)
        #params1_ = _wrap_params(params, params1)
        #log_r += _individual_log_ratio(dist, z, params0_, params1_)

    return log_r


def calculate_joint_score(trace, params=None):
    log_r = 0.
    for dist, z, params in _get_branchings(trace):
        params_ = _wrap_params(params, params)
        log_r += _individual_score(dist, z, params_)

    return log_r


def _wrap_params(original_params, new_params=None):
    if new_params is None:
        return None
    else:
        params_ = torch.zeros_like(original_params)
        params_[:, :] = new_params
    return params_


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


def _individual_log_prob(dist, z, params=None):
    if params is None:
        return dist.log_p(z)

    new_dist = type(dist)(params)
    return new_dist.log_prob(z)


def _individual_log_ratio(dist, z, params0=None, params1=None):
    log_p0 = _individual_log_prob(dist, z, params0)
    log_p1 = _individual_log_prob(dist, z, params1)
    return log_p0 - log_p1


def _individual_score(dist, z, params=None):
    raise NotImplementedError


def _get_param_names(distribution):
    param_names_unsorted = list(distribution.arg_constraints.keys())
    sig = inspect.signature(distribution.__init__)

    param_names_sorted = []
    for param in sig.parameters:
        if param in param_names_unsorted:
            param_names_sorted.append(param)
            param_names_unsorted.remove(param)

    return param_names_sorted + param_names_unsorted
