import torch
from torch.optim import Optimizer

class Muon(Optimizer):
    """A simple momentum-based optimizer.

    This is a lightweight implementation meant for demonstration and does not
    try to exactly replicate any particular research optimizer.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if momentum != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad)
                    grad = buf

                p.add_(grad, alpha=-lr)

        return loss
