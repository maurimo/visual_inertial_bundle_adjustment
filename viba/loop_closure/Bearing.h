/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <viba/problem/Types.h>

namespace visual_inertial_ba::loop {

// Unit ray of a pixel, such that the camera model's project() gives the pixel back within
// maxErrorPx. CameraModelParam::unprojectNoChecks alone is not enough: on the Fisheye624 test
// camera its round trip is off by up to ~100 px near the image border (TestLoopClosure,
// FivePoint.CameraRoundTrip), so it is only the start of a Gauss-Newton refinement on the sphere.
// Returns false if the refinement does not reach maxErrorPx.
inline bool bearingFromPixel(
    const CameraModelParam& cam,
    const Vec2& px,
    Vec3& f,
    double maxErrorPx = 0.01) {
  f = cam.unprojectNoChecks(px).normalized();
  if (!f.allFinite() || f.z() < 1e-3) {
    f = Vec3(px.x() - 0.5 * cam.imageWidth(), px.y() - 0.5 * cam.imageHeight(), 250.0).normalized();
  }
  for (int it = 0; it < 30; it++) {
    Vec2 p;
    Mat23 J;
    if (!cam.project(f, p, J)) {
      return false;
    }
    const Vec2 e = p - px;
    if (e.norm() < maxErrorPx) {
      return true;
    }
    // tangent basis at f
    const Vec3 a = std::abs(f.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
    const Vec3 b1 = f.cross(a).normalized(), b2 = f.cross(b1);
    Mat22 Jt;
    Jt.col(0) = J * b1;
    Jt.col(1) = J * b2;
    const Vec2 d = -Jt.colPivHouseholderQr().solve(e);
    const double step = std::min(1.0, 0.3 / std::max(d.norm(), 1e-12)); // at most 0.3 rad
    f = (f + step * (d[0] * b1 + d[1] * b2)).normalized();
  }
  Vec2 p;
  return cam.project(f, p) && (p - px).norm() < maxErrorPx;
}

} // namespace visual_inertial_ba::loop
