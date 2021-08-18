# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LBPSemanticDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of semantic dependency trees :cite:`wang-etal-2019-second`.
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
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
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

        logits = self.lbp(*scores, mask)
        marginals = logits.softmax(-1)[..., 1]

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask])

        return loss, marginals

    def lbp(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(
            -1, seq_len, seq_len)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        # mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(
        #     range(seq_len))).unsqueeze(-1)
        # mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(
        #     range(seq_len))).unsqueeze(-1)
        # [2, seq_len, seq_len, batch_size], (h->m)
        s_edge = torch.stack(
            (torch.zeros_like(s_edge), s_edge)).permute(0, 3, 2, 1)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0)

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge
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
            m_sib = torch.stack(
                (m.logsumexp(0), torch.stack(
                    (m[0], m[1] + s_sib)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_cop
            m_cop = torch.stack(
                (m.logsumexp(0), torch.stack(
                    (m[0], m[1] + s_cop)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_grd
            m_grd = torch.stack(
                (m.logsumexp(0), torch.stack(
                    (m[0], m[1] + s_grd)).logsumexp(0))).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_edge + (
                (m_sib + m_cop + m_grd).transpose(2, 3) * mask2o).sum(3)

        return q.permute(3, 2, 1, 0)


class MFVISemanticDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic dependency trees :cite:`wang-etal-2019-second`.
    """
    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter
        self.criterion = nn.CrossEntropyLoss()

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, null_idx, prd_idx, other_idxs, single_idxs, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                `s_label` (``[batch_size, seq_len, seq_len, n_labels]``)
                `s_sib_1` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples (belongs to one predicate). 
                `s_sib_2` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples (belongs to two predicates).
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

        logits = self.mfvi(*scores, mask, null_idx, prd_idx, other_idxs, single_idxs)
        marginals = F.softmax(logits, -1)

        if target is None:
            return marginals, logits
        
        loss = self.criterion(logits[mask], target[mask])
        # loss = F.binary_cross_entropy_with_logits(logits[mask],
        #                                           target[mask].float())

        return loss, marginals, logits

    def mfvi(self, s_label, s_sib_1, s_sib_2, s_cop, s_grd, mask, null_idx, prd_idx, other_idxs, single_idxs):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(
            -1, seq_len, seq_len)
        mask[:, 0] = 0
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        # mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        # mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [n_labels, seq_len, seq_len, batch_size], (h->m)
        s_label = s_label.permute(3, 2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib_1 = s_sib_1.permute(2, 1, 3, 0) * mask2o
        s_sib_2 = s_sib_2.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [n_labels, seq_len, seq_len, batch_size], (h->m)
        q = s_label
        n_labels = q.shape[0]

        for _ in range(self.max_iter):
            q = F.softmax(q, 0)  
            t_cop = ((1-q[null_idx]).transpose(0, 1).unsqueeze(0) * s_cop).sum(2)
            t_grd = ((1-q[null_idx]).unsqueeze(0) * s_grd).sum(2)
            new_q = q.clone()
            for i in other_idxs:
                new_q[i] = s_label[i] + (q[other_idxs[i]].unsqueeze(1) * s_sib_1
                                    + (1-(q[other_idxs[i]] + q[prd_idx] + q[null_idx])).unsqueeze(1) * s_sib_2).sum(2)
            for i in single_idxs:
                new_q[i] = s_label[i] + ((1-(q[prd_idx] + q[null_idx])).unsqueeze(1) * s_sib_2).sum(2)
            new_q += t_cop.unsqueeze(0).expand(n_labels, -1, -1, -1) + t_grd.unsqueeze(0).expand(n_labels, -1, -1, -1)
            new_q[null_idx] = s_label[null_idx]
            new_q[prd_idx] = s_label[prd_idx]
            q = new_q

            # # q(ij) = s(ij) + sum(q(ik)s^sib(ij,ik) + q(kj)s^cop(ij,kj) + q(jk)s^grd(ij,jk)), k != i,j
            # q = s_edge + (q.unsqueeze(1) * s_sib +
            #               q.transpose(0, 1).unsqueeze(0) * s_cop +
            #               q.unsqueeze(0) * s_grd).sum(2)

        return q.permute(3, 2, 1, 0)
