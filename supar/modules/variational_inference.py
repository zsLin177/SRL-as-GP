# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class LBPDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of dependency trees.
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
                Tuple of three tensors `s_arc`, `s_sib` and `s_grd`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
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

        logQ = self.lbp(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def lbp(self, s_arc, s_sib, s_grd, mask):
        batch_size, seq_len = mask.shape
        hs, ms = torch.stack(torch.where(mask.new_ones(seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        mask = mask.index_fill(1, hs.new_tensor(0), 1)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # log potentials of unary and binary factors
        # [seq_len, seq_len, batch_size], (h->m)
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0).masked_fill_(~mask2o, float('-inf'))
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0).masked_fill_(~mask2o, float('-inf'))

        # log beliefs
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_arc.new_zeros(seq_len, seq_len, batch_size).masked_fill_(~mask[0].unsqueeze(1), float('-inf'))
        # log messages of siblings
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = s_sib.new_zeros(seq_len, seq_len, seq_len, batch_size)
        # log messages of grand-parents
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        m_grd = s_grd.new_zeros(seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(2) - m_sib
            m_sib = torch.logaddexp(s_sib + m, m.logsumexp(0)).transpose(1, 2).log_softmax(0)
            m = q.unsqueeze(2) - m_grd
            m_grd = torch.logaddexp(s_grd + m, m.logsumexp(0)).transpose(1, 2).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_arc + ((m_sib + m_grd) * mask2o).sum(2)

        return q.permute(2, 1, 0).log_softmax(-1)


class MFVIDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of dependency trees (:cite:`wang-etal-2020-second`).
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
                Tuple of three tensors `s_arc`, `s_sib` and `s_grd`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
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

        logQ = self.mfvi(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def mfvi(self, s_arc, s_sib, s_grd, mask):
        batch_size, seq_len = mask.shape
        hs, ms = torch.stack(torch.where(mask.new_ones(seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        mask = mask.index_fill(1, hs.new_tensor(0), 1)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_arc.new_zeros(seq_len, seq_len, batch_size).masked_fill_(~mask[0].unsqueeze(1), float('-inf'))

        for _ in range(self.max_iter):
            q = q.softmax(0)
            # f(ij) = sum(q(ik)s^sib(ij,ik) + q(jk)s^grd(ij,jk)), k != i,j
            f = (q.unsqueeze(1) * s_sib + q.unsqueeze(0) * s_grd).sum(2)
            # q(ij) = s(ij) + f(ij)
            q = (s_arc + f).masked_fill_(~mask[0].unsqueeze(1), float('-inf'))

        return q.permute(2, 1, 0).log_softmax(-1)


class LBPSemanticDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of semantic dependency trees (:cite:`wang-etal-2019-second`).
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
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
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
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len, 2]``.
        """

        logQ = self.lbp(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def lbp(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # log potentials of unary and binary factors
        # [2, seq_len, seq_len, batch_size], (h->m)
        s_edge = torch.stack((torch.zeros_like(s_edge), s_edge)).permute(0, 3, 2, 1)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0)

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge.new_zeros(2, seq_len, seq_len, batch_size)
        # log messages of siblings
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = s_sib.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of co-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->c)
        m_cop = s_cop.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of grand-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->g)
        m_grd = s_grd.new_zeros(2, seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(3) - m_sib
            m_sib = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_sib)).logsumexp(0)))
            m_sib = m_sib.transpose(2, 3).log_softmax(0)
            m = q.unsqueeze(3) - m_cop
            m_cop = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_cop)).logsumexp(0)))
            m_cop = m_cop.transpose(2, 3).log_softmax(0)
            m = q.unsqueeze(3) - m_grd
            m_grd = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_grd)).logsumexp(0)))
            m_grd = m_grd.transpose(2, 3).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_edge + ((m_sib + m_cop + m_grd) * mask2o).sum(3)

        return q.permute(3, 2, 1, 0).log_softmax(-1)


class MFVISemanticDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic dependency trees (:cite:`wang-etal-2019-second`).
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
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
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
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len, 2]``.
        """

        logQ = self.mfvi(*(s.requires_grad_() for s in scores), mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = -logQ.gather(-1, target.unsqueeze(-1))[mask].sum() / mask.sum()

        return loss, marginals

    def mfvi(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0].unbind()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_edge = s_edge.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge.new_zeros(2, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.softmax(0)
            # f(ij) = sum(q(ik)s^sib(ij,ik) + q(kj)s^cop(ij,kj) + q(jk)s^grd(ij,jk)), k != i,j
            f = (q[1].unsqueeze(1) * s_sib + q[1].transpose(0, 1).unsqueeze(0) * s_cop + q[1].unsqueeze(0) * s_grd).sum(2)
            # q(ij) = s(ij) + f(ij)
            q = torch.stack((torch.zeros_like(q[0]), s_edge + f))

        return q.permute(3, 2, 1, 0).log_softmax(-1)
