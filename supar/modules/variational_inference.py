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
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len, 2]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
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
        marginals = log_beliefs.exp()

        if target is None:
            return marginals
        loss = -log_beliefs.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def belief_propagation(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        heads, dependents = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & heads.unsqueeze(-1).ne(heads.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & dependents.unsqueeze(-1).ne(dependents.new_tensor(range(seq_len))).unsqueeze(-1)
        # log potentials of unary and binary factors
        # [seq_len, seq_len, batch_size, 2], (h->m)
        p_edge = s_edge.permute(2, 1, 0, 3)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        p_sib = s_sib.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        p_cop = s_cop.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        p_grd = s_grd.permute(2, 1, 3, 0)

        # log beliefs
        # [seq_len, seq_len, batch_size, 2], (h->m)
        b = p_edge.new_zeros(seq_len, seq_len, batch_size, 2)
        # log messages of siblings
        # [seq_len, seq_len, seq_len, batch_size, 2], (h->m->s)
        m_sib = p_sib.new_zeros(seq_len, seq_len, seq_len, batch_size, 2)
        # log messages of co-parents
        # [seq_len, seq_len, seq_len, batch_size, 2], (h->m->c)
        m_cop = p_cop.new_zeros(seq_len, seq_len, seq_len, batch_size, 2)
        # log messages of grand-parents
        # [seq_len, seq_len, seq_len, batch_size, 2], (h->m->g)
        m_grd = p_grd.new_zeros(seq_len, seq_len, seq_len, batch_size, 2)

        for _ in range(self.max_iter):
            # b(ij) = p(ij) + sum(m(ik->ij)), min(i, j) < k < max(i, j)
            b = p_edge + ((m_sib + m_cop + m_grd) * mask2o.unsqueeze(-1)).sum(2)
            # m(ik->ij) = logsumexp(b(ik) - m(ij->ik) + p(ij->ik)) - m(ik->)
            m = b.unsqueeze(2) - m_sib
            m_sib = torch.stack((m.logsumexp(-1), torch.stack((m[..., 0], m[..., 1] + p_sib)).logsumexp(0)), -1)
            m_sib = m_sib.transpose(1, 2).log_softmax(-1)
            m = b.unsqueeze(2) - m_cop
            m_cop = torch.stack((m.logsumexp(-1), torch.stack((m[..., 0], m[..., 1] + p_sib)).logsumexp(0)), -1)
            m_cop = m_cop.transpose(1, 2).log_softmax(-1)
            m = b.unsqueeze(2) - m_grd
            m_grd = torch.stack((m.logsumexp(-1), torch.stack((m[..., 0], m[..., 1] + p_sib)).logsumexp(0)), -1)
            m_grd = m_grd.transpose(1, 2).log_softmax(-1)
        b = p_edge + ((m_sib + m_cop + m_grd) * mask2o.unsqueeze(-1)).sum(2)

        return b.permute(2, 1, 0, 3).log_softmax(-1)
