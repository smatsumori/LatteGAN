def get_grad_norm(params):
    """Get gradient norm.

    loss.backward()
    get_grad_norm(model.parameters())
    optimizer.step()

    Parameters
    ----------
    params : Iterator[torch.nn.parameter.Parameter]
        model.parameters()

    Returns
    -------
    grad_norm : torch.tensor
        shape (1,)
    """
    l2_norm = 0
    for param in params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            l2_norm += param_norm ** 2
    grad_norm = l2_norm ** (0.5)
    return grad_norm
