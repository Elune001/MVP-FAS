import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
from utils.visualization import visualize_attn
import math
class SlotAttention_PQTK(nn.Module):
    def __init__(self, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, texts):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype

        slots = inputs

        # slots = texts.repeat(b,1,1)# only text slot attention
        # normalize
        texts_norm = self.norm_input(texts.repeat(b,1,1))
        # For slot attention, forward inputs into FC layer
        k, v = self.to_k(texts_norm), self.to_v(texts_norm)

        # coarse to fine slot
        for _ in range(self.iters):
            slots_prev = slots

            # query: slot
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # compute similarity by q.matmul(k.transpose(-2,-1)) and divide sqrt(dim)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # dots = q.bmm(k.transpose(-2, -1)) * self.scale

            # ***** slot axis softmax *****, not a image tokens axis.
            attn = dots.softmax(dim=1) + self.eps

            # Make N axis summation of attention score [B, K, N] to 1 for weighted mean
            # These are normalize attention score by summation of the probability
            # that the image token N is assigned to the K-th slot
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # visualization code##
            # visualize_attn(attn, attn_shape=(b, int(math.sqrt(n)),int(math.sqrt(n)), 6),iter=_)
            ######################

            # attention_score.matmul(v)
            updates = torch.einsum('bjd,bij->bid', v, attn) # weighted mean
            # updates = attn.bmm(v)

            # inputs = update, state= slot_prev
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            # residual MLP (optional)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

