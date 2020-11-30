# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LoopyBeliefPropagation(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating partitions and marginals of semantic dependency trees.

    References:
        - Xinyu Wang, Jingxian Huang and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_..

    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_edge` and `s_sib`.
                `s_edge` (``[batch_size, seq_len, seq_len, 2]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                Tensors of gold-standard dependent-head pairs and dependent-head-sibling triples.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        s_edge, s_sib = scores
        logZ, marginals = self.belief_propagation(*(s.requires_grad_() for s in scores), mask)

        if target is None:
            return marginals
        edges, sibs = target
        s_edge = s_edge.gather(-1, edges.unsqueeze(-1))[mask]
        s_sib = s_sib[sibs.gt(0)]
        score = s_edge.sum() + s_sib.sum()
        loss = (logZ - score) / mask[:, 1].sum(-1)

        return loss, marginals

    def belief_propagation(self, s_edge, s_sib, mask):
        # [seq_len, seq_len, batch_size]
        mask = mask.permute(1, 2, 0)
        # [seq_len, seq_len, seq_len, batch_size]
        sib_mask = (mask.unsqueeze(0) & mask.unsqueeze(1)).permute(2, 1, 0, 3)
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
            # b(ij) = p(ij) + sum(m(ik->ij)), k
            b = (p_edge + (m_sib * sib_mask.unsqueeze(-1)).sum(2)) * mask.unsqueeze(-1)
            # m(ik->ij) = logsumexp(p(ij->ik) + b(ik) - m(ij->ik)) - m(ik->)
            m_sib = (p_sib + b.unsqueeze(1) - m_sib).logsumexp(0)
            m_sib = m_sib - m_sib.logsumexp(-1, True)
        return b[1, 0].logsumexp(-1).sum(), b.permute(2, 1, 0, 3)
