#!/usr/bin/env python3
"""sd_start.py <VI-BA open_loop_trajectory.csv> <first line of online_calibration.jsonl> <pose file: t_ns p q(x y z w)> <out csv>

A VI-BA starting trajectory from another estimator's poses (2026-11-26: SLAM Dunk's pose graph / batch on sequence_1_20).
The other trajectory's world and body frames are unknown: fitted as R_ours_imu = W R_other C (world rotation W, body
rotation C) and p_ours_imu = W p_other + t + W R_other b (body offset b), on all rows. Its poses are then written in
our world and our IMU frame at our rows (position linear, attitude slerp), the device pose through T_Device_Imu,
velocities from central differences; the other columns of the csv are kept."""
import sys, json
import numpy as np

def q2R(q):  # x y z w
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t); return np.array([(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, s/4])
    k = int(np.argmax(np.diag(R))); i, j = (k+1) % 3, (k+2) % 3
    s = 2 * np.sqrt(1 + R[k,k] - R[i,i] - R[j,j]); q = np.empty(4)
    q[k], q[i], q[j], q[3] = s/4, (R[i,k]+R[k,i])/s, (R[j,k]+R[k,j])/s, (R[j,i]-R[i,j])/s
    return q
def proj(M):
    U, _, Vt = np.linalg.svd(M); return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
def slerp(qa, qb, a):
    d = np.dot(qa, qb)
    if d < 0: qb, d = -qb, -d
    if d > 0.9995: q = qa + a * (qb - qa); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1-a)*th) * qa + np.sin(a*th) * qb) / np.sin(th)

csv, calib, other, out = sys.argv[1:5]
L = open(csv).read().splitlines(); head = L[0]; rows = [r.split(",") for r in L[1:] if r.strip()]
c = {n: i for i, n in enumerate(head.split(","))}
ip = [c[f"t{x}_odometry_device"] for x in "xyz"]; iq = [c[f"q{x}_odometry_device"] for x in "xyzw"]
iv = [c[f"device_linear_velocity_{x}_odometry"] for x in "xyz"]
tn = np.array([int(r[c["utc_timestamp_ns"]]) for r in rows]) if "utc_timestamp_ns" in c else None
T = json.loads(open(calib).read())["ImuCalibrations"][0]["T_Device_Imu"]
R_di = q2R(np.r_[T["UnitQuaternion"][1], T["UnitQuaternion"][0]]) if "UnitQuaternion" in T else None
t_di = np.array(T["Translation"])
if R_di is None: sys.exit("no rotation in T_Device_Imu")
Rd = np.array([q2R(np.array([float(r[i]) for i in iq])) for r in rows]); Pd = np.array([[float(r[i]) for i in ip] for r in rows])
Ri = Rd @ R_di; Pi = Pd + Rd @ t_di                                   # our IMU poses
A = np.loadtxt(other); to, po, Qo = A[:, 0], A[:, 1:4], A[:, 4:8]
k = np.clip(np.searchsorted(to, tn), 1, len(to) - 1); a = np.clip((tn - to[k-1]) / (to[k] - to[k-1]), 0, 1)
ok = (tn >= to[0]) & (tn <= to[-1])
Ro = np.array([q2R(slerp(Qo[k[n]-1], Qo[k[n]], a[n])) for n in range(len(tn))]); Po = po[k-1] + a[:, None] * (po[k] - po[k-1])
W, C = np.eye(3), np.eye(3)
for _ in range(20):
    W = proj(np.einsum("nij,nkj->ik", Ri[ok] @ C.T, Ro[ok]))
    C = proj(np.einsum("nji,njk->ik", Ro[ok], np.einsum("ij,njk->nik", W.T, Ri[ok])))
res = np.degrees(np.array([np.arccos(np.clip((np.trace((W @ Ro[n] @ C).T @ Ri[n]) - 1) / 2, -1, 1)) for n in np.where(ok)[0]]))
tilt = np.degrees(np.arccos(np.clip(W[2, 2], -1, 1)))
M = np.zeros((3 * ok.sum(), 6)); y = np.zeros(3 * ok.sum())
for m, n in enumerate(np.where(ok)[0]):
    M[3*m:3*m+3, :3] = np.eye(3); M[3*m:3*m+3, 3:] = W @ Ro[n]; y[3*m:3*m+3] = Pi[n] - W @ Po[n]
x, *_ = np.linalg.lstsq(M, y, rcond=None); tw, b = x[:3], x[3:]
if '--no-body-offset' in sys.argv:  # both are poses of the same IMU: the offset is 0 and only the world translation is fitted
    b = np.zeros(3); tw = (Pi[ok] - (W @ Po[ok].T).T).mean(0)
Rn = np.einsum("ij,njk->nik", W, Ro) @ C; Pn = (W @ Po.T).T + tw + np.einsum("ij,njk,k->ni", W, Ro, b)
print(f"fit: world rotation tilt {tilt:.3f} deg (yaw part {np.degrees(np.arctan2(W[1,0], W[0,0])):+.2f}), body rotation {np.degrees(np.arccos(np.clip((np.trace(C)-1)/2,-1,1))):.2f} deg, "
      f"body offset {np.round(b, 3)} m; attitude difference after the fit: median {np.median(res):.2f} deg, 95 % {np.percentile(res, 95):.2f}; "
      f"position difference to ours rms {np.sqrt(((Pn[ok] - Pi[ok]) ** 2).sum(1).mean()):.3f} m; rows outside the other trajectory: {(~ok).sum()}")
Rdn = Rn @ R_di.T; Pdn = Pn - Rdn @ t_di
td = tn * 1e-9; V = np.gradient(Pdn, td, axis=0)
with open(out, "w") as f:
    f.write(head + "\n")
    for n, r in enumerate(rows):
        if not ok[n]: f.write(",".join(r) + "\n"); continue
        r = list(r); q = R2q(Rdn[n])
        for j in range(3): r[ip[j]] = "%.10f" % Pdn[n, j]; r[iv[j]] = "%.10f" % V[n, j]
        for j in range(4): r[iq[j]] = "%.12f" % q[j]
        f.write(",".join(r) + "\n")
