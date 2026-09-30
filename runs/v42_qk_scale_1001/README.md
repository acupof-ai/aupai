# qk_scale over the v42 shared-KV arms (extracted 2026-10-01)

Every `qk_scale` line of arm b and arm d, lifted out of `runs/v42_arch_{b,d}_0930.log` on the pod,
because those logs are pod-only and a fact may not cite a path that is not in the repo. The
quantity is `max_h(q_rms_h) * kv_rms * softmax_scale` per layer, largest head, with the head mean
and the 24-layer median beside it (`v41f/optim.py` `qk_scale_report`, printed at the accum
boundary). It is the instrument the code already carried for exactly this question.

Reading: `facts/v41.json#v41.v42_shared_kv_logit_runaway_1001`. arm b runs to step 1100 only
because it was stopped there; arm d runs to 2000, its planned stop.
