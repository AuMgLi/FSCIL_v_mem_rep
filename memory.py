import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

USE_GPU = torch.cuda.is_available()


class MemoryK(nn.Module):

    def __init__(self, mem_size, key_dim, K, qry_dim=512, age_noise=6, alpha=0.1, is_norm=True):
        super().__init__()
        self.m_sz = mem_size
        self.key_dim = key_dim
        self.qry_dim = qry_dim
        self.K = K
        self.alpha = alpha
        self.is_norm = is_norm
        self.bs = self.n_c = self.n_hw = None

        m_keys = torch.randn((self.m_sz, self.key_dim))
        m_vals = torch.zeros((self.m_sz, ), dtype=torch.int) - 1
        m_ages = torch.zeros((self.m_sz, )) + float('inf')
        m_winner = torch.ones((self.m_sz, ), dtype=torch.int)
        age_noise = torch.randn([self.m_sz]) * age_noise
        self.register_buffer('m_keys', m_keys)
        self.register_buffer('m_vals', m_vals)
        self.register_buffer('m_ages', m_ages)
        self.register_buffer('m_winner', m_winner)
        self.register_buffer('age_noise', age_noise)
        # self.register_buffer('m_step', m_step)

        self.val_counter = {}
        self.n_classes = 0
        # self.usefulness = torch.zeros((self.m_sz, ), requires_grad=False) - 1  # for empty slots
        # self.qry_proj = nn.Linear(qry_dim, key_dim)  # LSTM?

    @torch.no_grad()
    def del_noisy_slots(self):
        """ 在新session开始时执行
        每轮至少能为下一个session空出m_sz / n_cls
        """
        counter = Counter(self.m_vals.cpu().numpy())
        self.n_classes = len(counter)
        avg_slots_p_cls = self.m_sz // self.n_classes
        for cls, n in counter.items():
            n_del = n - avg_slots_p_cls
            if cls == -1 or n_del <= 0:
                continue
            cls_slots_idxs = torch.where(torch.eq(self.m_vals, cls))[0]
            # assert len(cls_slots_idxs) == n
            # print(cls_slots_idxs)
            # _, useless_idxs = torch.topk(self.m_winner, k=n_del, largest=False)
            _, useless_idxs = torch.topk(self.m_ages[cls_slots_idxs], k=n_del)  # [n_del]
            useless_idxs = torch.gather(cls_slots_idxs, dim=0, index=useless_idxs)
            self.m_vals[useless_idxs] = -1
            self.m_ages[useless_idxs] = float('inf')
            self.m_winner[useless_idxs] = 1

    @torch.no_grad()
    def _update(self, query, target, correct_mask, nearest_pos_idx,
                top2_idxs, top2_vals, merge_correct_only=True):
        """
        :param query: [HW, key_dim]
        :param target: []
        :param correct_mask: [HW, K]
        :param nearest_pos_idx: [HW, 1]
        :param top2_idxs: [HW, 2]
        """
        # Merge similar winner and sub-winner
        nearest_idx = top2_idxs[:, 0]  # [HW]

        if self.K >= 2:
            second_nearest_idx = top2_idxs[:, 1]  # [HW]
            if merge_correct_only:
                idxs_to_merge = torch.where(torch.eq(correct_mask[:, :2].sum(dim=-1), 2))[0]  # [n_to_merge]
            else:
                nearest_val, second_nearest_val = top2_vals[:, 0], top2_vals[:, 1]  # [HW]
                idxs_to_merge = torch.where(torch.eq(nearest_val, second_nearest_val))[0]  # [n_to_merge]
            if len(idxs_to_merge) > 0:
                # print('idxs_to_merge:', idxs_to_merge)
                nearest_merge_idx = nearest_idx[idxs_to_merge]
                second_nearest_merge_idx = second_nearest_idx[idxs_to_merge]
                nearest_keys = torch.gather(  # [n_to_merge, key_dim]
                    self.m_keys, dim=0,
                    index=nearest_merge_idx.unsqueeze(-1).repeat(1, self.key_dim))
                second_nearest_keys = torch.gather(  # [n_to_merge, key_dim]
                    self.m_keys, dim=0,
                    index=second_nearest_merge_idx.unsqueeze(-1).repeat(1, self.key_dim))
                merged_keys = (nearest_keys + second_nearest_keys) / 2  # [n_to_merge, key_dim]  # Todo
                if self.is_norm:
                    merged_keys = F.normalize(merged_keys)
                self.m_keys[nearest_merge_idx] = merged_keys
                self.m_vals[second_nearest_merge_idx] = -1
                self.m_ages[second_nearest_merge_idx] = float('inf')
                self.m_winner[second_nearest_merge_idx] = 1

        nearest_pos_key = torch.gather(  # [HW, key_dim]
            self.m_keys, dim=0,
            index=nearest_pos_idx.repeat(1, self.key_dim))
        nearest_pos_winner = torch.gather(self.m_winner, dim=0, index=nearest_pos_idx.squeeze(1))  # [HW]

        # key_correct_upd = nearest_pos_key + (query / nearest_pos_winner.unsqueeze(1))  # [HW, key_dim]
        key_correct_upd = (nearest_pos_key + query) / 2
        if self.is_norm:
            key_correct_upd = F.normalize(key_correct_upd)

        m_ages_noised = self.m_ages + self.age_noise  # [m_sz]
        _, oldest_idx = torch.topk(m_ages_noised, k=self.n_hw, sorted=False)  # [HW]
        # _, useless_idx = torch.topk(self.usefulness, k=self.n_hw, largest=False, sorted=False)

        # Update age
        # age_plus1 = self.m_ages + 1
        # age_plus_w_step_decay = self.m_ages + (1 / (1 + 1e-3 * self.m_step))
        # self.m_ages.data = torch.where(
        #     torch.eq(self.m_vals, target),
        #     age_plus1,
        #     age_plus_w_step_decay
        # )
        self.m_ages.data = torch.where(
            torch.eq(self.m_vals, target),
            self.m_ages + 1,
            self.m_ages,
        )

        correct_lookup = correct_mask.sum(dim=-1).bool()  # [HW]
        nearest_pos_idx = nearest_pos_idx.squeeze(-1)  # [HW]
        upd_idxs = torch.where(
            correct_lookup,
            nearest_pos_idx,  # correct
            oldest_idx,  # incorrect
        )  # [HW]
        # print('upd_idxs', upd_idxs)
        upd_keys = torch.where(
            correct_lookup.unsqueeze(-1).repeat(1, self.key_dim),
            key_correct_upd,
            query,
        )  # [HW, key_dim]
        upd_winner = torch.where(
            correct_lookup,
            nearest_pos_winner + 1,
            torch.ones_like(nearest_pos_winner),
        )
        self.m_keys[upd_idxs] = upd_keys
        self.m_vals[upd_idxs] = target.int()
        self.m_ages[upd_idxs] = 1
        self.m_winner[upd_idxs] = upd_winner

    def forward(self, queries: torch.Tensor, targets):
        """
        :param queries: [bs, C, H, W]
        :param targets: [bs]
        :return: memory loss
        """
        memory_loss = torch.zeros([])
        if USE_GPU:
            memory_loss = memory_loss.cuda()
        self.bs, self.n_c, n_h, n_w = queries.size()
        self.n_hw = n_h * n_w
        assert self.n_hw <= self.m_sz and self.n_c == self.key_dim  # channel transformed in encoder
        fetched_vals = torch.zeros([self.bs], dtype=torch.int)

        # Project queries vector into memory key
        queries = queries.view(self.bs, self.n_c, self.n_hw).transpose(1, 2)  # [bs, HW, C(key_dim)]
        # bshw = self.bs * self.n_hw
        # queries = self.qry_proj(queries.reshape(bshw, self.n_c)).view(
        #     self.bs, self.n_hw, self.key_dim)  # [bs, HW, key_dim]
        if self.is_norm:
            queries = F.normalize(queries, dim=-1)

        for i in range(self.bs):
            qry_img = queries[i]  # [HW, key_dim]
            tgt_img = targets[i]  # []

            m_keys_t = self.m_keys.transpose(0, 1).contiguous()  # [key_dim, m_sz]
            if self.is_norm:  # cosine distance
                glb_sims = torch.mm(qry_img, m_keys_t)  # [HW, m_sz]
            else:  # euclidean distance
                # queries_ext = queries.
                # glb_sims = F.pairwise_distance()
                glb_sims = torch.mm(qry_img, m_keys_t)  # [HW, m_sz]

            # Nearest neighbor
            _, top2_idxs = torch.topk(glb_sims, k=2, dim=-1)  # [HW, 2]
            # nearest_idx, second_nearest_idx = top2_idxs[:, 0], top2_idxs[:, 1]  # [HW]
            m_vals_ext = self.m_vals.repeat(self.n_hw, 1)  # [HW, m_sz]
            top2_vals = torch.gather(m_vals_ext, dim=1, index=top2_idxs)  # [HW, 2]
            nearest_val, second_nearest_val = top2_vals[:, 0], top2_vals[:, 1]  # [HW]
            # Hard k-nn via H*W features
            counter = Counter(nearest_val.cpu().numpy())
            fetched_vals[i] = torch.tensor(counter.most_common(1)[0][0], dtype=torch.int)

            # Top-K
            topk_sims, topk_idxs = torch.topk(glb_sims, k=self.K, dim=-1)  # [HW, K]
            topk_vals = torch.gather(m_vals_ext, dim=1, index=topk_idxs)  # [HW, K]
            safety_pos_idx = torch.where(torch.eq(self.m_vals, tgt_img))[0]
            if len(safety_pos_idx) == 0:  # No slots with same value
                correct_mask = topk_vals == tgt_img  # [HW, K]
                assert correct_mask.sum() == 0
                nearest_pos_idx = torch.zeros([self.n_hw, 1], dtype=torch.long)
                if USE_GPU:
                    nearest_pos_idx = nearest_pos_idx.cuda()
                incorrect_mask = ~correct_mask  # [HW, K]
                nearest_neg_sim, nearest_neg_idx = torch.topk(topk_sims * incorrect_mask, k=1)
                loss = torch.relu(nearest_neg_sim + self.alpha).mean()
            else:
                safety_pos_idx = safety_pos_idx[0]
                safety_pos_sim = glb_sims[:, [safety_pos_idx]]  # [HW, 1]
                topk_sims = torch.cat([topk_sims, safety_pos_sim], dim=1)  # [HW, K+1]
                topk_vals = torch.cat(
                    [topk_vals, tgt_img.int().unsqueeze(0).repeat(self.n_hw, 1)], dim=1)
                # Calculate memory loss
                correct_mask = torch.eq(topk_vals, tgt_img)  # [HW, K+1]
                incorrect_mask = ~correct_mask
                # Nearest positive via nearest-K slots
                nearest_pos_sim, nearest_pos_idx = torch.topk(topk_sims * correct_mask, k=1)  # [HW, 1]
                if nearest_pos_idx == self.K:
                    nearest_pos_idx = nearest_pos_idx - 1  # avoid index out of range
                correct_mask, _ = correct_mask.split([self.K, 1], dim=-1)  # [HW, K]
                # print('nearest_pos_sim:', nearest_pos_sim.mean())
                nearest_neg_sim, nearest_neg_idx = torch.topk(topk_sims * incorrect_mask, k=1)
                loss = torch.relu(nearest_neg_sim - nearest_pos_sim + self.alpha).mean()  # triplet loss
            memory_loss += loss

            if self.training:  # Update memory
                nearest_pos_idx = torch.gather(topk_idxs, dim=1, index=nearest_pos_idx)  # [HW, 1]
                self._update(qry_img, tgt_img, correct_mask, nearest_pos_idx, top2_idxs, top2_vals)
        if USE_GPU:
            fetched_vals = fetched_vals.cuda()
        return fetched_vals, memory_loss


class Memory(nn.Module):

    def __init__(self, mem_size, key_dim, tau=0.8, qry_dim=512, age_noise=6, alpha=0.1, need_norm=True):
        super().__init__()
        self.m_sz = mem_size
        self.key_dim = key_dim
        self.qry_dim = qry_dim
        self.tau = tau
        self.alpha = alpha
        self.need_norm = need_norm
        self.bs = self.n_c = self.n_hw = None

        m_keys = torch.randn((self.m_sz, self.key_dim))
        if self.need_norm:
            m_keys = F.normalize(m_keys)
        m_vals = torch.zeros((self.m_sz,), dtype=torch.int) - 1
        m_ages = torch.zeros((self.m_sz,)) + float('inf')
        m_winner = torch.ones((self.m_sz,), dtype=torch.int)
        age_noise = torch.randn([self.m_sz]) * age_noise
        self.register_buffer('m_keys', m_keys)
        self.register_buffer('m_vals', m_vals)
        self.register_buffer('m_ages', m_ages)
        self.register_buffer('m_winner', m_winner)
        self.register_buffer('age_noise', age_noise)

        self.val_counter = {}
        self.n_classes = 0
        self.m_centroids = None

    @torch.no_grad()
    def del_noisy_slots(self):
        counter = Counter(self.m_vals.cpu().numpy())
        self.n_classes = len(counter)
        avg_slots_p_cls = self.m_sz // self.n_classes
        for cls, n in counter.items():
            n_del = n - avg_slots_p_cls + 1
            if cls == -1 or n_del <= 1:
                continue
            cls_slots_idxs = torch.where(torch.eq(self.m_vals, cls))[0]
            _, useless_idxs = torch.topk(self.m_ages[cls_slots_idxs], k=n_del)  # [n_del]
            useless_idxs = torch.gather(cls_slots_idxs, dim=0, index=useless_idxs)
            useless_keys = torch.gather(self.m_keys, dim=0,
                                        index=useless_idxs.unsqueeze(-1).repeat(1, self.key_dim))
            useless_keys_mean = useless_keys.mean(dim=0)
            self.m_keys[useless_idxs[-1]] = useless_keys_mean
            self.m_vals[useless_idxs[:-1]] = -1
            self.m_ages[useless_idxs[:-1]] = float('inf')
            # self.m_winner[useless_idxs] = 1

    @torch.no_grad()
    def upd_centroids(self):
        # Calculate Centroids
        n_classes, _ = torch.topk(self.m_vals, dim=0, k=1)
        m_centroid_keys = []
        m_centroid_vals = []
        for v in range(n_classes.item() + 1):
            m_keys_v = self.m_keys[torch.eq(self.m_vals, v)]
            if len(m_keys_v) == 0:
                continue
            v_key_mean = m_keys_v.mean(dim=0)
            m_centroid_keys.append(v_key_mean)
            m_centroid_vals.append(v)
        m_centroid_keys = torch.stack(m_centroid_keys, dim=0)
        m_centroid_vals = torch.tensor(m_centroid_vals, dtype=torch.long)
        if USE_GPU:
            m_centroid_keys = m_centroid_keys.cuda()
            m_centroid_vals = m_centroid_vals.cuda()
        if self.need_norm:
            m_centroid_keys = F.normalize(m_centroid_keys)
        # print('centroids keys:', m_centroid_keys.shape)  # Todo
        # print('centroid vals:', m_centroid_vals)
        self.m_centroids = (m_centroid_keys, m_centroid_vals)
        print("==> Memory centroids updated.")

    def forward(self, queries: torch.Tensor, targets, upd_memory=True, use_centroid=False):
        """
        :param queries: [bs, C, H, W]
        :param targets: [bs]
        :param upd_memory:
        :param use_centroid:
        :return: memory loss
        """
        memory_loss = torch.zeros([])
        if USE_GPU:
            memory_loss = memory_loss.cuda()
        self.bs, self.n_c, n_h, n_w = queries.size()
        self.n_hw = n_h * n_w
        assert self.n_hw <= self.m_sz and self.n_c == self.key_dim  # channel transformed in encoder

        # Project queries vector into memory key
        queries = queries.view(self.bs, self.n_c, self.n_hw).transpose(1, 2)  # [bs, HW, C(key_dim)]
        # bshw = self.bs * self.n_hw
        # queries = self.qry_proj(queries.reshape(bshw, self.n_c)).view(
        #     self.bs, self.n_hw, self.key_dim)  # [bs, HW, key_dim]
        if self.need_norm:
            queries = F.normalize(queries, dim=-1)

        fetched_vals = torch.zeros([self.bs], dtype=torch.int)
        for i in range(self.bs):
            qry_img = queries[i]  # [HW, key_dim]
            tgt_img = targets[i]  # []

            m_keys_t = self.m_keys.transpose(0, 1).contiguous()  # [key_dim, m_sz]
            m_vals_ext = self.m_vals.repeat(self.n_hw, 1)  # [HW, m_sz]
            # Calc global similarity
            if self.need_norm:  # cosine distance
                glb_sims = torch.mm(qry_img, m_keys_t)  # [HW, m_sz]
            else:  # euclidean distance
                # queries_ext = queries.
                # glb_sims = F.pairwise_distance()
                glb_sims = torch.mm(qry_img, m_keys_t)  # [HW, m_sz]

            if use_centroid:  # NMC
                centroid_keys, centroid_vals = self.m_centroids  # [n_class, key_dim]
                centroid_keys_t = centroid_keys.transpose(0, 1).contiguous()  # [key_dim, n_class]
                centroid_sims = torch.mm(qry_img, centroid_keys_t)  # [HW, n_class]

                _, nearest_idxs = torch.topk(centroid_sims, k=1, dim=-1)  # [HW, 1]
                nearest_val = torch.gather(centroid_vals.repeat(self.n_hw, 1), dim=1, index=nearest_idxs)  # [HW, 1]
                # Hard k-nn via H*W features
                counter = Counter(nearest_val.squeeze(-1).cpu().numpy())
                fetched_vals[i] = torch.tensor(counter.most_common(1)[0][0], dtype=torch.int)
            else:  # Nearest neighbor
                _, nearest_idxs = torch.topk(glb_sims, k=1, dim=-1)  # [HW, 1]
                nearest_val = torch.gather(m_vals_ext, dim=1, index=nearest_idxs)  # [HW, 1]
                # Hard k-nn via H*W features
                counter = Counter(nearest_val.squeeze(-1).cpu().numpy())
                fetched_vals[i] = torch.tensor(counter.most_common(1)[0][0], dtype=torch.int)

            correct_mask = torch.eq(m_vals_ext, tgt_img)  # [HW, m_sz]
            incorrect_mask = ~correct_mask
            pos_sims = glb_sims * correct_mask  # [HW, m_sz]
            neg_sims = glb_sims * incorrect_mask
            nearest_pos_sim, nearest_pos_idx = torch.topk(pos_sims, k=1, dim=1)  # [HW, 1]
            nearest_neg_sim, nearest_neg_idx = torch.topk(neg_sims, k=1, dim=1)
            # print('nearest_neg_sim:', nearest_neg_sim, 'nearest_neg_idx:', nearest_neg_idx)
            if correct_mask.sum() == 0:
                loss = torch.relu(nearest_neg_sim + self.alpha).mean()
            else:
                loss = torch.relu(nearest_neg_sim - nearest_pos_sim + self.alpha).mean()  # triplet loss
            memory_loss += loss

            if self.training and upd_memory:  # Update memory
                with torch.no_grad():
                    # Merge similar winner and sub-winner
                    top2_pos_sims, top2_pos_idxs = torch.topk(pos_sims, k=2, dim=1)  # [HW, 2]
                    idxs_to_merge = torch.where(
                        (top2_pos_sims[:, 0] > 0.1) &
                        (top2_pos_sims[:, 0] - top2_pos_sims[:, 1] < (1 - self.tau) / 2))[0]  # [n_to_merge]
                    if len(idxs_to_merge) > 0:
                        # print('idxs_to_merge:', idxs_to_merge)
                        nearest_merge_idx = top2_pos_idxs[:, 0]
                        second_nearest_merge_idx = top2_pos_idxs[:, 1]
                        nearest_keys = torch.gather(  # [n_to_merge, key_dim]
                            self.m_keys, dim=0,
                            index=nearest_merge_idx.unsqueeze(-1).repeat(1, self.key_dim))
                        second_nearest_keys = torch.gather(  # [n_to_merge, key_dim]
                            self.m_keys, dim=0,
                            index=second_nearest_merge_idx.unsqueeze(-1).repeat(1, self.key_dim))
                        merged_keys = (nearest_keys + second_nearest_keys) / 2  # [n_to_merge, key_dim]  # Todo
                        if self.need_norm:
                            merged_keys = F.normalize(merged_keys)
                        self.m_keys[nearest_merge_idx] = merged_keys
                        self.m_vals[second_nearest_merge_idx] = -1
                        self.m_ages[second_nearest_merge_idx] = float('inf')
                        # self.m_winner[second_nearest_merge_idx] = 1

                    nearest_pos_key = torch.gather(self.m_keys, dim=0,  # [HW, key_dim]
                                                   index=nearest_pos_idx.repeat(1, self.key_dim))
                    # nearest_pos_winner = torch.gather(self.m_winner, dim=0,
                    #                                   index=nearest_pos_idx.squeeze(1))

                    key_correct_upd = (nearest_pos_key + qry_img) / 2
                    if self.need_norm:
                        key_correct_upd = F.normalize(key_correct_upd)

                    m_ages_noised = self.m_ages + self.age_noise
                    _, oldest_idx = torch.topk(m_ages_noised, k=self.n_hw, sorted=False)
                    self.m_ages.data = torch.where(
                        torch.eq(self.m_vals, tgt_img),
                        self.m_ages + 1,
                        self.m_ages,
                    )
                    # self.m_ages.data += 1

                    correct_lookup = nearest_pos_sim.squeeze(1) > self.tau  # [HW]
                    # assert correct_lookup.size() == (self.n_hw,)
                    upd_idxs = torch.where(
                        correct_lookup,
                        nearest_pos_idx.squeeze(1),
                        oldest_idx,
                    )
                    upd_keys = torch.where(
                        correct_lookup.unsqueeze(-1).repeat(1, self.key_dim),
                        key_correct_upd,
                        qry_img,
                    )
                    # upd_winner = torch.where(
                    #     correct_lookup,
                    #     nearest_pos_winner + 1,
                    #     torch.ones_like(nearest_pos_winner),
                    # )
                    self.m_keys[upd_idxs] = upd_keys
                    self.m_vals[upd_idxs] = tgt_img.int()
                    self.m_ages[upd_idxs] = 1
                    # self.m_winner[upd_idxs] = upd_winner

                    # self._update(qry_img, tgt_img, correct_mask, nearest_pos_idx, top2_idxs, top2_vals)
        if USE_GPU:
            fetched_vals = fetched_vals.cuda()
        return fetched_vals, memory_loss


def debug():
    m = Memory(mem_size=5, qry_dim=3, key_dim=3, tau=0.7, need_norm=True)
    # print(m.m_ages)
    qry = torch.randn([2, 3, 1, 1])
    # print('qry:', qry)
    tgt = torch.tensor(data=[0, 1], dtype=torch.long)
    # m.m_ages[2] = 100
    print(m.m_keys.data)
    print('val:', m.m_vals)
    print('age:', m.m_ages)
    print('winner:', m.m_winner)
    m(qry, tgt)
    m.del_noisy_slots()
    print(m.m_keys)
    print('val:', m.m_vals)
    print('age:', m.m_ages)
    print('winner:', m.m_winner)


if __name__ == '__main__':
    debug()
