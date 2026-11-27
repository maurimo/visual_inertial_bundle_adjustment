# Sensor model: parasitic light sensitivity (row time) and IMU sample timing

Aria profile only (`--profile aria`, chosen automatically for Aria inputs). Options: `--pls-scale` (1 in the Aria
profile; 0: off), `--pls-eps` (1/89), `--pls-t0-ms` (0.35), `--pls-row-us` (19), `--pls-smear-px` (0.8),
`--pls-smear-corner-px` (5), `--pls-dir` (default: the input folder); `--imu-half-sample` (on in the Aria profile),
`--accel-later-ms` (0, per device). Code: `SingleSessionAdapter::plsInit` / `plsUpdate`
(`viba/single_session/VisualFactors.cpp`), `RollingShutterVisualFactor*` (`viba/problem/VisualFactor.cpp`).

## Parasitic light: a global-shutter image that is not taken at one time

The SLAM cameras of Aria are global shutter: every pixel is exposed during the same exposure time $t_e$. But after
the exposure, while the image is read out row by row, the stored charge keeps collecting light with a small
sensitivity $\varepsilon$ ("parasitic light sensitivity"). Row $r$ is read at $w = t_0 + r \cdot t_{\text{row}}$
after the end of the exposure, so it receives light with weight 1 over the exposure and weight $\varepsilon$ over
the following $w$. The centroid in time of what the row saw is later than the centre of the exposure by

$$ D(r) = \frac{\varepsilon\, w\,(t_e + w)}{2\,(t_e + \varepsilon\, w)},\qquad w = t_0 + r\,t_{\text{row}} $$

It matters when the exposure is short: with $t_e$ of a fraction of a millisecond the ghost is a sizeable part of the
signal, and the bottom rows of a moving image appear later than the top ones, like a small rolling shutter whose
size depends on the exposure.

**Smear.** The ghost is not a sharp copy: it is smeared along the image motion over $s = u\,w/\ell$ correlation
lengths ($u$ the image speed in px/s, $\ell$ the correlation length of the feature). A matcher with a feature of
autocorrelation $\exp(-x^2/2\ell^2)$ puts it at $\delta = x\,u\,w$ with
$x = (g/s^2)\,(1 - e^{-s^2(1-2x)/2})$, $g = \varepsilon w / t_e$ the weight of the ghost; $x \to g/(2(1+g))$ (the
centroid) for $s \to 0$ and $g/s^2$ for $s \gg 1$. The model's time is multiplied by the attenuation
$x / x_0$. Corner (KLT) tracks and photometric tracks have different correlation lengths (5 px and 0.8 px).

**Parameters**, fitted on the exposure study and confirmed by VI-BA (below): $t_0 = 0.35$ ms (not 1 ms),
$\varepsilon = 1/89$ (not 1/65), $t_{\text{row}} = 19\ \mu$s, 480 rows.

**Implementation.** Each observation with a known exposure gets a time shift $D$ (times `--pls-scale`, with the
smear attenuation), recomputed from the current trajectory with the rolling-shutter data (before each optimization
call). The observation then uses the rolling-shutter visual factor: the camera pose at the shifted time comes from the
inertial samples around the frame, the rig's velocity and gravity, so the factor also depends on the rig's
**velocity variable** (with exact Jacobians). Inputs: `expo_cam<k>.csv` (lines `t_ns,exposure_s`, the image nearest
in time within 2 ms) and optionally `obs_kinds.csv` (one line per observation, `frame,id,camera,kind`, kind 0 =
corner); without the exposures the model is off whatever the profile. The velocity in every observation makes the
reduced system denser (9 instead of 6 dimensions per rig in each point's clique): with time order the factor of
R_11_5cp grows from 1.3 to 1.7 GB and a factorization from 15 to 28 s ([elimination_order.md](elimination_order.md);
with AMD it was 4.3 GB and 3 min).

## IMU sample timing

- **Half-sample hold** (`--imu-half-sample`). Aria's factory IMU time offsets are instantaneous sample times, while
  VI-BA's preintegration holds each sample constant over the interval that *ends* at its timestamp. The offset to use
  is the factory one minus half a sample period (0.499 ms on imu-right at 1 kHz), which is also what MPS's online
  calibration reports. Inputs whose converters already shift the IMU stamps (TUM-VI / EuRoC, point samples) use the
  generic profile, where this is off.
- **Accelerometer against gyroscope** (`--accel-later-ms`): the factory delay between the two is off on some devices
  (+0.37 ms on one pair of glasses); per device, 0 by default.

## Results

VI-BA estimates both IMU time offsets per 5 s calibration window (factory values as prior and start). R_11_5cp, mean
over the windows, ms:

| | gyroscope | accelerometer | accel - gyro | cost |
|---|---|---|---|---|
| factory | 4.146 | 3.121 | -1.025 | |
| prediction (half-sample hold + corrections) | 3.647 | 2.256 | -1.391 | |
| VI-BA, row-time model off | 4.213 (+-0.295) | 2.846 | -1.367 | 106 446 |
| VI-BA, row-time model, old parameters (1 ms, 1/65) | 3.456 (+-0.084) | 2.066 | -1.390 | 100 604 |
| VI-BA, row-time model, fitted parameters (0.35 ms, 1/89) | **3.674** (+-0.039) | **2.292** | -1.382 | **99 636** |

The free estimate lands 0.03-0.04 ms from the prediction (old parameters 0.19 ms off, factory 0.47 ms), whatever the
prior centre. Without the row-time model the gyro offset absorbs the mean image delay (0.56 ms), which is why the
factory value looked right. On a second device (sequence_1_19) accel - gyro is -1.32 ms (factory -1.46, predicted
-1.28): with the corrections both devices show the same delay.

**Found by the optimizer too.** A rolling-shutter readout of 9.1 ms written for these global-shutter cameras and
estimated freely settles at 1.7-1.9 ms top to bottom, varying 0.8-2.6 ms along the recording, with the cost 7.5 %
below the global-shutter optimum: the size, sign and variability that the exposure study gives for parasitic light.

**Scores** (LaMAria control points). On a filter's observations of R_11_5cp (3.2 M observations), two passes: cost
219 109 -> 191 971 (-12.4 %), score 98.8 / 98.5 -> 99.3 / 99.4 % (pass 1 / 2), ATE 0.194 -> 0.162 m. But the model
does not help everywhere: it raises the heading rate, and where the rate left over is positive it costs score
(R_12_10cp); on later observations of R_11_5cp it scores 97.4 % against 98.1 % without it, and the recordings with
very short exposures (sequence_3_*) are still open. Whether it stays on in the Aria profile is decided per data set
with `--pls-scale`.
