"""
Don't-be-a-Hero VAE Model
"""
from torch import nn
from torch.autograd import Function, Variable
import os
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class VAE(nn.Module):
    def __init__(self, z_dim, k=10, bn=False, vq_coef=1, commit_coef=0.5, p_in=3, **kwargs):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(p_in, z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_dim, z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            ResBlock(z_dim, z_dim, bn=bn),
            nn.BatchNorm2d(z_dim),
            ResBlock(z_dim, z_dim, bn=bn),
            nn.BatchNorm2d(z_dim),
        )
        self.decoder = nn.Sequential(
            ResBlock(z_dim, z_dim),
            nn.BatchNorm2d(z_dim),
            ResBlock(z_dim, z_dim),
            nn.ConvTranspose2d(z_dim, z_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(z_dim, p_in, kernel_size=4, stride=2, padding=1),
        )
        self.z_dim = z_dim
        self.emb = NearestEmbed(k, z_dim)
        self.vq_coef = vq_coef

        self.encoder_avg = nn.Sequential(self.encoder, nn.AvgPool2d(16))

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return {
            "x_hat": self.decode(z_q),
            "z_e": z_e,
            "emb": emb,
            "argmin": argmin
        }

    def sample(self, size):
        sample = torch.randn(size, self.z_dim, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.z_dim, self.f, self.f)).cpu()

def vae_loss(x, y, output, vq_coef=1, commit_coef=0.5):
    x_hat = output["x_hat"]
    z_e = output["z_e"]
    emb = output["emb"]
    argmin = output["argmin"]

    mse = F.mse_loss(x_hat, x)
    vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
    commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))
    return mse + vq_coef*vq_loss + commit_coef*commit_loss


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)
