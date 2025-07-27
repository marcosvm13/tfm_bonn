import torch
import torch.nn as nn
import torch.nn.functional as F


class GateBonn(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=2000, output_dim=10, tau=5,
                 beta_alpha=1.2, beta_beta=3.0, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.logits_pi = nn.Parameter(torch.zeros(hidden_dim))

    def sample_z(self, batch_size):
        u = torch.rand((batch_size, self.hidden_dim), device=self.device)
        gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        pi = torch.sigmoid(self.logits_pi)
        logits = torch.log(pi + 1e-9) - torch.log(1 - pi + 1e-9)
        logits = logits.unsqueeze(0).expand(batch_size, -1)
        z = torch.sigmoid((logits + gumbel_noise) / self.tau)
        return z, pi

    def forward(self, x):
        z, pi = self.sample_z(x.size(0))
        h = torch.sigmoid(self.fc1(x)) * z
        out = self.fc2(h)
        return out, pi, z

    def estimate_mutual_information(self, h, labels, temperature=0.1):
        h = F.normalize(h, dim=1)
        B = h.size(0)
        sim = torch.matmul(h, h.T) / temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(h.device)
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=h.device)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / \
            (mask.sum(dim=1) + 1e-9)
        return mean_log_prob_pos.mean()

    def kl_regularization(self, pi):
        alpha_q = 1.0 + pi
        beta_q = 1.0 + (1.0 - pi)
        alpha_p = torch.tensor(self.beta_alpha, device=pi.device)
        beta_p = torch.tensor(self.beta_beta, device=pi.device)
        term1 = torch.lgamma(alpha_p + beta_p) - \
            torch.lgamma(alpha_p) - torch.lgamma(beta_p)
        term2 = torch.lgamma(alpha_q) + torch.lgamma(beta_q) - \
            torch.lgamma(alpha_q + beta_q)
        term3 = (alpha_q - alpha_p) * (torch.digamma(alpha_q) -
                                       torch.digamma(alpha_q + beta_q))
        term4 = (beta_q - beta_p) * (torch.digamma(beta_q) -
                                     torch.digamma(alpha_q + beta_q))
        return (term1 + term2 + term3 + term4).sum()
