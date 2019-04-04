import torch
from torch.autograd import grad
import pyro
from pyro import poutine
import inspect

from benchmark import Simulator


class PyroSimulator(Simulator):
    """ Pyro simulator interface """

    def forward(self, inputs):
        raise NotImplementedError

    def trace(self, inputs):
        return pyro.poutine.trace(self.forward).get_trace(inputs)

    def augmented_data(self, inputs, inputs_num, inputs_den):
        """
        Forward pass of the simulator that also calculates the joint likelihood ratio and the joint score, as defined
        in arXiv:1805.12244.

        Args:
            inputs (torch.Tensor): Values of the parameters used for sampling. Have shape (n_batch, n_parameters). The
                                   joint score is also evaluated at these parameters.
            inputs_num (torch.Tensor): Values of the parameters used for the numerator of the joint likelihood ratio.
            inputs_den (torch.Tensor): Values of the parameters used for the denominator of the joint likelihood ratio.

        Returns:
            outputs (torch.Tensor): Generated data (observables), sampled from `p(outputs | inputs)`.
            joint_score (torch.Tensor): Joint score `grad_inputs log p(outputs, latents | inputs)`.
            joint_score (torch.Tensor): Joint log likelihood ratio
                                        `log (p(outputs, latents | inputs_num) / p(outputs, latents | inputs_den))`.

        """
        trace = self.trace(inputs)

        x = self._calculate_x(trace)
        joint_score = self._calculate_joint_score(trace, inputs)
        joint_likelihood_ratio = self._calculate_joint_likelihood_ratio(trace, inputs_num, inputs_den)

        return x, joint_score, joint_likelihood_ratio

    def _replayed_trace(self, original_trace, inputs):
        if inputs is None:
            return original_trace

        return poutine.trace(poutine.replay(self.forward, original_trace)).get_trace(inputs)

    @staticmethod
    def _calculate_x(trace):
        node = trace.nodes["_RETURN"]
        x = node["value"]
        return x

    def _calculate_joint_log_prob(self, trace):
        log_p = 0.
        for dist, z, _ in self._get_branchings(trace):
            log_p = log_p + dist.log_prob(z)
        return log_p

    def _calculate_joint_score(self, trace, params):
        params.requires_grad = True
        score = 0.
        for dist, z, _ in self._get_branchings(trace):
            log_p = dist.log_prob(z)
            score = score + grad(
                log_p,
                params,
                grad_outputs=torch.ones_like(log_p.data),
                only_inputs=True,
                create_graph=False,
            )
        return score

    def _calculate_joint_likelihood_ratio(self, trace, params_num, params_den):
        trace_num = self._replayed_trace(trace, params_num)
        trace_den = self._replayed_trace(trace, params_den)

        log_p_num = self._calculate_joint_log_prob(trace_num)
        log_p_den = self._calculate_joint_log_prob(trace_den)

        return log_p_num - log_p_den

    def _get_branchings(self, trace):
        for key in trace.nodes:
            if key in ["_INPUT", "_RETURN"]:
                continue
            node = trace.nodes[key]

            dist = node["fn"]
            z = node["value"]

            params = []
            for param_name in self._get_param_names(dist):
                param = getattr(dist, param_name)
                if len(param.size()) == 1:
                    param = param.view(-1, 1)
                params.append(param)
            params = torch.cat(params, 1)

            yield dist, z, params

    @staticmethod
    def _get_param_names(distribution):
        param_names_unsorted = list(distribution.arg_constraints.keys())
        sig = inspect.signature(distribution.__init__)

        param_names_sorted = []
        for param in sig.parameters:
            if param in param_names_unsorted:
                param_names_sorted.append(param)
                param_names_unsorted.remove(param)

        return param_names_sorted + param_names_unsorted
