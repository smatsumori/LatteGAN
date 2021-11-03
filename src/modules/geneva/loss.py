import torch
import torch.nn.functional as F
from torch import autograd


class HingeAdversarial():
    """Hinge Adversarial Loss as used in
    https://arxiv.org/pdf/1802.05957.pdf"""
    @staticmethod
    def discriminator(real, fake, wrong=None, wrong_weight=None):
        """Discriminator loss term

        Parameters
        ----------
        real : torch.tensor
            shape (B,)
            D(x)
        fake : torch.tensor
            shape (B,)
            D(G(z))
        wrong : torch.tensor, optional
            shape (B,), by default None
        wrong_weight : int, optional
            weight for wrong discriminator, by default None

        Returns
        -------
        loss : torch.tensor
            shape (1,)
        """

        l_real = F.relu(1. - real).mean()
        l_fake = F.relu(1. + fake).mean()

        if wrong is None:
            loss = l_real + l_fake
        else:
            l_wrong = F.relu(1. + wrong).mean()
            loss = l_real + wrong_weight * \
                l_wrong + (1. - wrong_weight) * l_fake
        return loss

    @staticmethod
    def generator(fake):
        """Generator loss term

        Parameters
        ----------
        fake : torch.tensor
            shape (B,)
            D(G(x))

        Returns
        -------
        loss : torch.tensor
            shape (1,)
        """
        return -fake.mean()


def gradient_penalty(discriminator_out, data_point):
    """Regularizing GAN training by penalizing
    gradients for real data only, such that it does
    not produce orthogonal gradients to the data
    manifold at equilibrium.
    Follows:
    Which Training Methods for GANs do actually Converge? eq.(9)
    Args:
        - discriminator_out: output logits from the
        discriminator
        - data_point: real data point (x_real).
    Returns:
        reg: regularization value.
    """
    batch_size = data_point.size(0)
    grad_dout = autograd.grad(outputs=discriminator_out.sum(),
                              inputs=data_point,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]

    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == data_point.size())
    reg = grad_dout2.view(batch_size, -1).sum(1).mean()

    return reg


def kl_penalty(mu, logvar):
    """KL-Divergence for train ConditioningAugmentor

    KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    mu : torch.tensor
        shape (B, D)
    logvar : torch.tensor
        shape (B, D)

    Returns
    -------
    KLD : torch.tensor
        shape (1,)
        Negative KL Divergence
    """
    KLD_element = 1 + logvar - mu.pow(2) - logvar.exp()
    KLD = -0.5 * torch.mean(KLD_element)
    return KLD
