# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LoopyBeliefPropagation(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals of semantic dependency trees.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_..

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_edge` and `s_sib`.
                `s_edge` (``[batch_size, seq_len, seq_len, 2]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        log_beliefs = self.belief_propagation(*(s.requires_grad_() for s in scores), mask)

        if target is None:
            return log_beliefs.exp()
        loss = -log_beliefs.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, log_beliefs.exp()

    def belief_propagation(self, s_edge, s_sib, mask):
        _, seq_len, _ = mask.shape
        heads, dependents = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        sib_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).permute(0, 1, 2, 3)
        sib_mask = sib_mask & heads.unsqueeze(-1).ne(heads.new_tensor(range(seq_len))).unsqueeze(-1)
        sib_mask = sib_mask & dependents.unsqueeze(-1).ne(dependents.new_tensor(range(seq_len))).unsqueeze(-1)
        # log potentials for unary and binary factors, i.e., edges and siblings
        # [seq_len, seq_len, batch_size, 2], (h->m)
        p_edge = s_edge.permute(2, 1, 0, 3)
        # [2, seq_len, seq_len, seq_len, batch_size, 2], (h->m->s)
        p_sib = (s_sib.unsqueeze(-1).unsqueeze(-1) * s_sib.new_tensor([[0, 0], [0, 1]])).permute(4, 2, 1, 3, 0, 5)

        # log beliefs
        # [seq_len, seq_len, batch_size, 2], (h->m)
        b = torch.zeros_like(p_edge)
        # log messages for siblings
        # [seq_len, seq_len, seq_len, batch_size, 2], (h->m->s)
        m_sib = torch.zeros_like(p_sib[0])

        for _ in range(self.max_iter):
            # b(ij) = p(ij) + sum(m(ik->ij)), min(i, j) < k < max(i, j)
            b = p_edge + (m_sib * sib_mask.unsqueeze(-1)).sum(2)
            # m(ik->ij) = logsumexp(p(ij->ik) + b(ik) - m(ij->ik)) - m(ik->)
            m_sib = ((p_sib + b.unsqueeze(1) - m_sib).logsumexp(0).transpose(1, 2)).log_softmax(-1)
        b = p_edge + (m_sib * sib_mask.unsqueeze(-1)).sum(2)

        return b.permute(2, 1, 0, 3).log_softmax(-1)
