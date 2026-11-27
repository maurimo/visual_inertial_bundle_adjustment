#!/usr/bin/env python3
"""lc_split_viba_input.py <session_observations.csv> <out dir> [--gap 27] [--max-pairs 20]

Splits a VI-BA input's observations (e.g. SLAM Dunk's batch input converted by aria_slamdunk_to_viba.py, whose
revisit observations are merged into the same point ids) for the loop-closed VI-BA (2026-11-27):
  <out>/session_observations.csv  the points WITHOUT a gap of more than --gap seconds (rows as they are)
  <out>/loop_tracks.csv           the points WITH such a gap, all their observations: track,visit,t_us,cam,u,v
                                  (visit = segment between gaps, in time order)
  <out>/lc_matches.csv            correspondences between visits of the loop points, for the pose-graph stage:
                                  t_a_us,cam_a,t_b_us,cam_b,u_a,v_a,u_b,v_b (consecutive visits, at most --max-pairs
                                  observation pairs per pair of visits)
Times in the input are microseconds (capture_timestamp_ns column holds us, as VI-BA's converters write it)."""
import os, sys
import numpy as np
a = sys.argv[1:]
src, out = a[0], a[1]
gap = float(a[a.index("--gap") + 1]) if "--gap" in a else 27.0
maxp = int(a[a.index("--max-pairs") + 1]) if "--max-pairs" in a else 20
os.makedirs(out, exist_ok=True)
lines = open(src).read().splitlines(); head = lines[0]; body = lines[1:]
d = np.array([[float(x) for x in l.split(",")[:5]] for l in body])
pid, t, cam, u, v = d[:, 0].astype(np.int64), d[:, 1].astype(np.int64), d[:, 2].astype(np.int64), d[:, 3], d[:, 4]
o = np.lexsort((t, pid)); ps, ts = pid[o], t[o]
same = np.r_[False, ps[1:] == ps[:-1]]
brk = same & (np.r_[0, np.diff(ts)] > gap * 1e6)
loop_ids = np.unique(ps[brk])
is_loop = np.isin(pid, loop_ids)
with open(os.path.join(out, "session_observations.csv"), "w") as f:
    f.write(head + "\n")
    for l, k in zip(body, is_loop):
        if not k: f.write(l + "\n")
# visits
visit = np.zeros(len(ps), np.int64); cur = 0
for i in range(len(ps)):
    if i == 0 or ps[i] != ps[i - 1]: cur = 0
    elif brk[i]: cur += 1
    visit[i] = cur
inv = np.empty_like(o); inv[o] = np.arange(len(o))
vis = visit[inv]
sel = np.where(is_loop)[0]; sel = sel[np.lexsort((t[sel], pid[sel]))]
with open(os.path.join(out, "loop_tracks.csv"), "w") as f:
    f.write("track,visit,t_us,cam,u,v\n")
    for i in sel: f.write(f"{pid[i]},{vis[i]},{t[i]},{cam[i]},{u[i]:.4f},{v[i]:.4f}\n")
nm = 0
with open(os.path.join(out, "lc_matches.csv"), "w") as f:
    f.write("t_a_us,cam_a,t_b_us,cam_b,u_a,v_a,u_b,v_b\n")
    start = 0
    while start < len(sel):
        end = start
        while end < len(sel) and pid[sel[end]] == pid[sel[start]]: end += 1
        rows = sel[start:end]; vv = vis[rows]
        for va in range(vv.max()):
            A, B = rows[vv == va], rows[vv == va + 1]
            pairs = [(x, y) for x in A for y in B][:maxp]
            for x, y in pairs:
                f.write(f"{t[x]},{cam[x]},{t[y]},{cam[y]},{u[x]:.4f},{v[x]:.4f},{u[y]:.4f},{v[y]:.4f}\n"); nm += 1
        start = end
n_vis = len(np.unique(np.stack([pid[sel], vis[sel]], 1), axis=0))
print(f"{src}: {len(np.unique(pid))} points, {len(pid)} observations; loop points (gap > {gap:.0f} s): {len(loop_ids)} with "
      f"{is_loop.sum()} observations in {n_vis} visits; base: {len(np.unique(pid[~is_loop]))} points, {(~is_loop).sum()} observations; "
      f"{nm} correspondences for the pose graph")
