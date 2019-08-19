from .kernel import Kernel
from ..lazy import BlockDiagLazyTensor
from .. import delazify
import torch

class BlockDiagTaskKernel(Kernel):
    def __init__(self, base_kernels):
        """
        Given data (x1, x2), applies a set of base kernels to the data and returns a lazy tensor equivalent to a
        block diagonal matrix which each of these base kernel applications.

        Using the 2 base kernel case as an example, this kernel computes a lazy tensor representing the following
        covariance matrix:
                     [base_kernels[0](x1, x2)             0          ]
        K(x1, x2) =  [         0              base_kernels[1](x1, x2)]
        """
        super().__init__()
        self.base_kernels = base_kernels

    def forward(self, x1, x2, diag=False):
        # b list of length n vectors in diag mode, b list of n x m matrices in nondiag mode
        base_kernel_results = [base_kern.forward(x1, x2, diag=diag) for base_kern in self.base_kernels]
        # Concatenate results to a `b x n` (possibly lazy) tensor for diag mode, or a `b x n x m` tensor for non diag.
        cat_res = gpytorch.cat(base_kernel_results)

        if diag:
            # In diag mode, just concatenate all base diagonals.
            return delazify(cat_res)
        else:
            # Form a BlockDiagLazyTensor -- converts a b x n x m lazy tensor to a bn x bm block diagonal lazy tensor.
            res = BlockDiagLazyTensor(cat_res)
            return res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this kernel returns an
        (n*num_base_kernels) x (m*num_base_kernels) covariance matrix.
        """
        return len(self.base_kernels)
