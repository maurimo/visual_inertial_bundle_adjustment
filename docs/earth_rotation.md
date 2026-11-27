# Sensor model: rotation of the Earth

`--earth-rate-up <rad/s>` (Aria profile: 5.365e-5, the value for LaMAria's location; generic profile: 0, off).
Code: `InertialFactor` (`viba/problem/InertialFactor.cpp`), `SingleSessionProblem::setEarthRateWorld`.

## The model

A gyroscope measures the rotation of the body with respect to inertial space, while VI-BA's world frame is fixed to
the Earth, which turns at $\Omega = 7.292\cdot10^{-5}$ rad/s. Over a preintegration interval $\Delta t$ the measured
relative rotation is therefore

$$ R_{\text{prev,next}}^{\text{measured}} = R_{\text{prev},w}\;\exp(\omega_E\,\Delta t)\;R_{w,\text{next}} $$

with $\omega_E$ the Earth's angular velocity in world coordinates. VI-BA's world is gravity aligned but its heading
is arbitrary, so only the **vertical component** can be modelled without knowing north:
$\omega_E = (0, 0, \Omega \sin\varphi)$ at latitude $\varphi$. In Zurich (47.4 deg N, LaMAria):
$7.29\cdot10^{-5}\,\sin(47.37^\circ) = 5.37\cdot10^{-5}$ rad/s $= 0.184$ deg/min. The horizontal component
needs the direction of north and is not modelled. The value depends on the location of the recording: give
`--earth-rate-up` explicitly for other places.

## Results

Score % (LaMAria control points), pass 1 / pass 2, older observations:

| recording | without | with |
|---|---|---|
| sequence_1_20 | 62.0 / 61.8 | 65.7 / 65.6 |
| sequence_1_19 | 71.5 / 84.3 | 75.1 / 88.7 |
| R_11_5cp | 97.3 / 96.7 | 97.9 / 97.3 |
| R_12_10cp | 59.7 / 64.1 | 57.6 / 61.4 |

The heading rate of the solution moves by about -0.02 deg/min, 9-15 % of the Earth's vertical rate (the gyro bias had
absorbed the rest), as in the filter. It helps where the remaining heading rate is positive and costs where it is
negative; one sample per recording, so the scores are an indication. The model is physically right and cheap (a
constant rotation per inertial factor), so it is on in the Aria profile.
