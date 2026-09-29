"""Incremental CED inference engine (v41_pivot.md Steps 6-7): prefill skip + bounded replay.

Training recomputes the whole sequence every token (scripts/base_generate.py). This engine
runs one new position per layer per step and keeps two caches:

  * SWA ring KV per layer, width n_win (128).
  * CSA2 global entries, one per completed m-token (8) block:
      - encoder CSA2 layers 2-5 learn entries from THEIR OWN pooled k/v (Full mode);
      - decoder layers 6-11 project entries from the encoder boundary state H_6 through
        the per-layer unshared W_KV/W_Z (CED Eq.1).

Two decoder-prefill replay modes:
  exact   - replay 1 + n_dec*(n_win-1) positions so every top-layer window key near the
            boundary is computed with an untruncated SWA history; per-step logits match a
            whole-sequence forward at bf16 kernel precision.
  bounded - replay the last n_win positions only (tech report §2.2/§3.2.2): SWA is
            truncated inside the replay segment, global entries still come from the FULL
            H_6. An approximation by design; scored against exact, never expected to match.

Single sequence (B=1, one document): the per-document masks in model.py collapse to
"same document everywhere", leaving only the causal window and complete-and-past block
visibility. Attention arithmetic mirrors the CPU branch of model.csa2_window_flash and
CompressedSparseAttention._forward_csa2.
"""
import torch
import torch.nn.functional as F


class CEDCache:
    def __init__(self, model, mode="bounded"):
        if mode not in ("exact", "bounded"):
            raise ValueError(f"mode must be exact|bounded, got {mode}")
        self.m = model
        cfg = model.cfg
        self.cfg = cfg
        self.mode = mode
        self.L = int(cfg.layers)
        self.split = int(cfg.ced_enc_layers)
        self.n_win = int(cfg.csa2_n_win)
        self.mblk = int(cfg.csa2_m)
        self.topk = int(cfg.csa2_top_k)
        self.ih = int(cfg.csa2_indexer_heads)
        self.di = int(cfg.csa2_indexer_dim)
        self.H = int(cfg.heads)
        self.gh = self.H // self.ih
        self.hd = int(cfg.d) // self.H
        self.blocks = model.blocks
        self.reset()

    def reset(self):
        # SWA ring per layer: lists of single-position (H,hd) rows in arrival order.
        self.swa_k = {i: [] for i in range(self.L)}
        self.swa_v = {i: [] for i in range(self.L)}
        # Encoder CSA2 layers: finalized entry (H,1,hd) lists + running m-block rows.
        self.enc_kc = {i: [] for i in range(self.split)}
        self.enc_vc = {i: [] for i in range(self.split)}
        self.enc_part_k = {i: [] for i in range(self.split)}
        self.enc_part_v = {i: [] for i in range(self.split)}
        # Decoder CSA2 layers: finalized projected entries.
        self.dec_kc = {i: [] for i in range(self.split, self.L)}
        self.dec_vc = {i: [] for i in range(self.split, self.L)}
        self.h6_all = []          # H_6 row (d,) at every absolute position
        self.h6_part = []         # rows of the not-yet-complete H_6 m-block
        self.t = -1

    def _push_ring(self, i, k1, v1):
        self.swa_k[i].append(k1)
        self.swa_v[i].append(v1)
        if len(self.swa_k[i]) > self.n_win:
            self.swa_k[i].pop(0)
            self.swa_v[i].pop(0)

    def _ring(self, i):
        ks = self.swa_k[i]
        if not ks:
            z = next(self.m.parameters()).new_zeros((0, self.H, self.hd))
            return z, z
        return torch.stack(ks, 0), torch.stack(self.swa_v[i], 0)

    # ----------------------------------------------------------------- attention join
    def _entry_select(self, csa, xq, kc, vis):
        """Indexer hard top-k. xq: (Tq,d) layer-input rows; kc (H,NB,hd); vis (Tq,NB).
        Returns sel (Tq,H,NB). Mirrors _forward_csa2's mask-before-topk."""
        Tq, NB = xq.shape[0], kc.shape[1]
        iq = csa.indexer_q(xq).view(Tq, self.ih, self.di)                 # (Tq,ih,di)
        kc_g = kc.view(self.ih, self.gh, NB, self.hd).mean(1)            # (ih,NB,hd)
        ik = (kc_g @ csa.ik_weight)                                     # (ih,NB,di)
        isc = torch.einsum("qid,ind->qin", iq, ik) * (self.di ** -0.5)  # (Tq,ih,NB)
        isc = isc.masked_fill(~vis[:, None], float("-inf"))
        kk = min(self.topk, NB)
        topk_idx = isc.topk(kk, dim=-1).indices                         # (Tq,ih,kk)
        sel = torch.zeros(Tq, self.ih, NB, dtype=torch.bool, device=kc.device)
        sel.scatter_(-1, topk_idx, True)
        sel &= vis[:, None]
        return sel.repeat_interleave(self.gh, dim=1)                    # (Tq,H,NB)

    def _swa_qe(self, swa, q, wk, wv, win_mask):
        """PureSWA for Tq query rows against W window keys. Matches PureSWA's CPU fallback
        in model.py: softmax and the value matmul stay in the INPUT dtype (bf16), not fp32 --
        upcasting here diverges from the trained kernel by ~1e-2 at bf16. win_mask (Tq,W)."""
        qh = q.transpose(0, 1)                                        # (H,Tq,hd)
        kw = wk.transpose(0, 1)
        sc = (qh @ kw.transpose(-1, -2) * swa.scale).masked_fill(~win_mask[None], float("-inf"))
        y = torch.softmax(sc, -1) @ wv.transpose(0, 1)
        return y.transpose(1, 2)                                     # (Tq,H,hd)

    # ----------------------------------------------------------------- block finalizers
    def _learned_entries(self, csa, k_rows, v_rows):
        """compress_k/v over one m-block of own k/v; rows in position order -> (H,1,hd)."""
        ks = torch.stack(k_rows, 0).permute(1, 0, 2).reshape(self.H, self.mblk * self.hd)
        vs = torch.stack(v_rows, 0).permute(1, 0, 2).reshape(self.H, self.mblk * self.hd)
        kc = csa.compress_k(ks).reshape(self.H, 1, self.hd)
        vc = csa.compress_v(vs).reshape(self.H, 1, self.hd)
        return kc, vc

    def _project_h6_entry(self, i, h6_rows):
        # The MEAN must match _ced_kv_from_enc's arithmetic exactly: bf16 scatter_add
        # accumulates rows in POSITION order, then divides by the real row count. torch's
        # mean() uses a different (pairwise) reduction; at bf16 the two differ by ~0.1 in
        # the projected keys and that difference is the whole prefill gap on a trained
        # model. Sequential adds reproduce scatter_add's order bit for bit.
        csa = self.blocks[i].mixer.csa
        hb = h6_rows[0].clone()
        for r in h6_rows[1:]:
            hb = hb + r
        hb = hb / len(h6_rows)
        kc = csa.w_kv(hb).view(self.H, self.hd)
        vc = csa.w_z(hb).view(self.H, self.hd)
        if csa.kc_norm:
            kc = F.rms_norm(kc, (self.hd,))
        return kc[:, None, :], vc[:, None, :]

    def _dec_entries(self, i):
        """Finalized projected entries plus the in-progress H_6 tail block. Mirrors
        model._ced_kv_from_enc: a tail block is a MEAN over its real rows and is visible
        to a query at or past its (current) last position. Returns kc,vc (H,NB,hd) and
        blk_last (NB,)."""
        kc = torch.cat(self.dec_kc[i], 1) if self.dec_kc[i] else \
            next(self.m.parameters()).new_zeros((self.H, 0, self.hd))
        NB = kc.shape[1]
        last = (torch.arange(NB, device=kc.device) + 1) * self.mblk - 1
        if self.h6_part:
            tkc, tvc = self._project_h6_entry(i, self.h6_part)
            kc = torch.cat([kc, tkc], 1)
            vc = torch.cat([
                torch.cat(self.dec_vc[i], 1) if self.dec_vc[i] else
                tkc.new_zeros((self.H, 0, self.hd)), tvc], 1)
            last = torch.cat([last, torch.tensor(
                [NB * self.mblk + len(self.h6_part) - 1], device=kc.device)])
        else:
            vc = torch.cat(self.dec_vc[i], 1) if self.dec_vc[i] else \
                kc.new_zeros((self.H, 0, self.hd))
        return kc, vc, last

    def _enc_entries(self, i, csa):
        """Finalized learned entries plus the in-progress own-k/v tail block. The learned
        reducer sees a ZERO-PADDED m-block (entries_per_doc), so the tail entry is computed
        by padding the part rows, not by changing the linear. Returns kc,vc and blk_last."""
        z = next(self.m.parameters())
        kc = torch.cat(self.enc_kc[i], 1) if self.enc_kc[i] else z.new_zeros((self.H, 0, self.hd))
        NB = kc.shape[1]
        last = (torch.arange(NB, device=kc.device) + 1) * self.mblk - 1
        pk = self.enc_part_k[i]
        if pk:
            pad = self.mblk - len(pk)
            krows = pk + [z.new_zeros((self.H, self.hd)) for _ in range(pad)]
            vrows = self.enc_part_v[i] + [z.new_zeros((self.H, self.hd)) for _ in range(pad)]
            tkc, tvc = self._learned_entries(csa, krows, vrows)
            kc = torch.cat([kc, tkc], 1)
            vc0 = torch.cat(self.enc_vc[i], 1) if self.enc_vc[i] else z.new_zeros((self.H, 0, self.hd))
            vc = torch.cat([vc0, tvc], 1)
            last = torch.cat([last, torch.tensor(
                [NB * self.mblk + len(pk) - 1], device=kc.device)])
        else:
            vc = torch.cat(self.enc_vc[i], 1) if self.enc_vc[i] else z.new_zeros((self.H, 0, self.hd))
        return kc, vc, last

    # ----------------------------------------------------------------- decode: one layer, one pos
    def _enc_layer_step(self, i, x1, t):
        blk = self.blocks[i]
        mx = blk.mixer
        xn = blk.n1(x1)
        q, k, v, gate = mx._proj_qkv(xn, pos=torch.tensor([[t]], device=x1.device))
        q1, k1, v1 = q[0], k[0], v[0]                                # (1,H,hd)
        self._push_ring(i, k1[0], v1[0])
        if mx.swa is not None:
            wk, wv = self._ring(i)
            y = self._swa_qe(mx.swa, q1, wk, wv,
                             torch.ones(1, wk.shape[0], dtype=torch.bool, device=x1.device))
        else:
            csa = mx.csa
            self.enc_part_k[i].append(k1[0])
            self.enc_part_v[i].append(v1[0])
            if len(self.enc_part_k[i]) == self.mblk:                  # complete: finalize, reset
                kc0, vc0 = self._learned_entries(csa, self.enc_part_k[i], self.enc_part_v[i])
                self.enc_kc[i].append(kc0)
                self.enc_vc[i].append(vc0)
                self.enc_part_k[i], self.enc_part_v[i] = [], []
            kc, vc, last = self._enc_entries(i, csa)
            vis = (last <= t)[None]
            sel = self._entry_select(csa, xn, kc, vis) if kc.shape[1] else \
                torch.zeros(1, self.H, 0, dtype=torch.bool, device=x1.device)
            wk, wv = self._ring(i)
            wm = torch.ones(1, wk.shape[0], dtype=torch.bool, device=x1.device)
            y = self._join_seg(csa, q1, kc, vc, sel, wk, wv, wm)
        x2 = x1 + mx.o(y.reshape(1, 1, -1) * torch.sigmoid(gate))
        return x2 + blk.ffn(blk.n2(x2))

    def _dec_layer_seg(self, i, x, seg_pos, win_mask, kc, vc, blk_last):
        """Vectorized decoder layer over Tq replay rows. Global entries (incl. the H_6
        tail block) are fixed from the full prompt; win_mask truncates SWA inside the
        segment (its left edge is the bounded replay start)."""
        blk = self.blocks[i]
        mx = blk.mixer
        csa = mx.csa
        xn = blk.n1(x)
        q, k, v, gate = mx._proj_qkv(xn, pos=seg_pos[None])
        Tq = x.shape[1]
        keep = k[0, max(0, Tq - self.n_win):]                          # rows for later decode
        self.swa_k[i] = [r for r in keep.unbind(0)]
        self.swa_v[i] = [r for r in v[0, max(0, Tq - self.n_win):].unbind(0)]
        if kc.shape[1]:
            vis = blk_last[None, :] <= seg_pos[:, None]              # (Tq,NB)
            sel = self._entry_select(csa, xn[0], kc, vis)
        else:
            sel = torch.zeros(Tq, self.H, 0, dtype=torch.bool, device=x.device)
        y = self._join_seg(csa, q[0], kc, vc, sel, k[0], v[0], win_mask)
        x = x + mx.o(y.reshape(1, Tq, -1) * torch.sigmoid(gate))
        return x + blk.ffn(blk.n2(x))

    def _join_seg(self, csa, q, kc, vc, sel, wk, wv, win_mask):
        """Single- or multi-query join of selected entries and the SWA window, copied line
        for line from the CPU (use_flash=False) branch of model.csa2_window_flash -- the path
        the gated model takes on CPU. That keeps the incremental attention bit-identical to a
        whole-sequence forward. q (Tq,H,hd), sel (Tq,H,NB) hard selection, wk/wv (W,H,hd),
        win_mask (Tq,W). Returns (Tq,H,hd)."""
        Tq, H, hd = q.shape
        wd = torch.float64 if q.dtype == torch.float64 else torch.float32
        qh = q.transpose(0, 1)                                        # (H,Tq,hd)
        e_mask = sel.permute(1, 0, 2)                                # (H,Tq,NB)
        has_e = e_mask.any(-1)
        se = (qh @ kc.transpose(-1, -2) * csa.scale).masked_fill(~e_mask, float("-inf"))
        le = torch.logsumexp(se.to(wd), -1)
        le = torch.where(has_e, le, torch.full_like(le, float("-inf")))
        pe = torch.nan_to_num(torch.softmax(se, -1), nan=0.0) * has_e.unsqueeze(-1)
        # ste's forward value IS the hard selector, so multiply the selected entry output by it.
        oe = (pe * e_mask.to(pe.dtype)) @ vc                         # (H,Tq,hd)
        kw = wk.transpose(0, 1)                                      # (H,W,hd)
        sw = (qh @ kw.transpose(-1, -2) * csa.scale).masked_fill(
            ~win_mask[None], float("-inf"))
        lw = torch.logsumexp(sw.to(wd), -1)
        ow = torch.softmax(sw, -1).to(wd) @ wv.transpose(0, 1).to(wd)
        mm = torch.maximum(le, lw)
        ae = torch.where(torch.isfinite(le), (le - mm).exp(), torch.zeros_like(le))
        aw = (lw - mm).exp()
        den = ae + aw
        ce = torch.where(den > 0, ae / den, torch.zeros_like(ae))
        cw = torch.where(den > 0, aw / den, torch.zeros_like(aw))
        y = (oe.to(wd) * ce.unsqueeze(-1) + ow * cw.unsqueeze(-1))
        return y.to(q.dtype).transpose(0, 1)                        # (Tq,H,hd)

    def _dec_layer_step(self, i, x1, t):
        blk = self.blocks[i]
        mx = blk.mixer
        csa = mx.csa
        xn = blk.n1(x1)
        q, k, v, gate = mx._proj_qkv(xn, pos=torch.tensor([[t]], device=x1.device))
        q1, k1, v1 = q[0], k[0], v[0]
        self._push_ring(i, k1[0], v1[0])
        kc, vc, last = self._dec_entries(i)
        if kc.shape[1]:
            vis = (last <= t)[None]
            sel = self._entry_select(csa, xn, kc, vis)
        else:
            sel = torch.zeros(1, self.H, 0, dtype=torch.bool, device=x1.device)
        wk, wv = self._ring(i)
        wm = torch.ones(1, wk.shape[0], dtype=torch.bool, device=x1.device)
        y = self._join_seg(csa, q1, kc, vc, sel, wk, wv, wm)
        x2 = x1 + mx.o(y.reshape(1, 1, -1) * torch.sigmoid(gate))
        return x2 + blk.ffn(blk.n2(x2))

    # ----------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill(self, ids):
        self.reset()
        device = self.m.tok.weight.device
        idx = ids.view(1, -1).to(device)
        P = idx.shape[1]
        x = self.m.tok(idx)

        # encoder: full vectorized pass (exact), capturing each layer's k/v for the cache.
        for i in range(self.split):
            mx = self.blocks[i].mixer
            cap = {}
            mx._cap = cap
            x = self.blocks[i](x)
            mx._cap = None
            kf, vf = cap["k"][0], cap["v"][0]                      # (P,H,hd)
            self.swa_k[i] = [r for r in kf[max(0, P - self.n_win):].unbind(0)]
            self.swa_v[i] = [r for r in vf[max(0, P - self.n_win):].unbind(0)]
            if mx.swa is None:
                csa = mx.csa
                NBp = P // self.mblk
                if NBp:
                    kk = kf[:NBp * self.mblk].reshape(NBp, self.mblk, self.H, self.hd)
                    vv = vf[:NBp * self.mblk].reshape(NBp, self.mblk, self.H, self.hd)
                    bk = kk.permute(0, 2, 1, 3).reshape(-1, self.H, self.mblk * self.hd)
                    bv = vv.permute(0, 2, 1, 3).reshape(-1, self.H, self.mblk * self.hd)
                    kc = csa.compress_k(bk).reshape(NBp, self.H, 1, self.hd)
                    vc = csa.compress_v(bv).reshape(NBp, self.H, 1, self.hd)
                    self.enc_kc[i] = [r for r in kc.unbind(0)]
                    self.enc_vc[i] = [r for r in vc.unbind(0)]
                self.enc_part_k[i] = [r for r in kf[NBp * self.mblk:].unbind(0)]
                self.enc_part_v[i] = [r for r in vf[NBp * self.mblk:].unbind(0)]
        h_enc = x
        self.h6_all = [r for r in h_enc[0].unbind(0)]
        NBp = P // self.mblk
        C = NBp * self.mblk
        self.h6_part = self.h6_all[C:]
        for i in range(self.split, self.L):                         # entries from FULL H_6
            for b in range(NBp):
                kc, vc = self._project_h6_entry(i, self.h6_all[b * self.mblk:(b + 1) * self.mblk])
                self.dec_kc[i].append(kc)
                self.dec_vc[i].append(vc)

        # bounded/exact decoder replay over an H_6 suffix.
        n_dec = self.L - self.split
        s = (max(0, P - self.n_win) if self.mode == "bounded"
             else max(0, P - 1 - n_dec * (self.n_win - 1)))
        Tq = P - s
        seg = h_enc[:, s:P]
        seg_pos = torch.arange(s, P, device=device)
        ar = torch.arange(Tq, device=device)
        win_mask = torch.ones(Tq, Tq, dtype=torch.bool, device=device).tril()
        win_mask &= (ar[:, None] - ar[None, :]) < self.n_win
        out = seg
        for i in range(self.split, self.L):
            dkc, dvc, dlast = self._dec_entries(i)   # finalized blocks + H_6 tail, shared by replay rows
            out = self._dec_layer_seg(i, out, seg_pos, win_mask, dkc, dvc, dlast)
        self.t = P - 1
        return self._logits(out[0, -1])

    # ----------------------------------------------------------------- decode
    @torch.no_grad()
    def step(self, token_id):
        device = self.m.tok.weight.device
        self.t += 1
        t = self.t
        x = self.m.tok(torch.tensor([[token_id]], device=device))
        for i in range(self.split):
            x = self._enc_layer_step(i, x, t)
        h6 = x[0, 0]
        self.h6_all.append(h6)
        self.h6_part.append(h6)
        if len(self.h6_part) == self.mblk:
            for i in range(self.split, self.L):
                kc, vc = self._project_h6_entry(i, self.h6_part)
                self.dec_kc[i].append(kc)
                self.dec_vc[i].append(vc)
            self.h6_part = []
        for i in range(self.split, self.L):
            x = self._dec_layer_step(i, x, t)
        return self._logits(x[0, -1])

    def _logits(self, hidden):
        # hidden is one row (d,); unsqueeze to (1,d) -> lm_logits (1,V) -> (V,).
        return self.m.lm_logits(self.m.norm(hidden.unsqueeze(0)))[0]
