/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/Types.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Where the wide-baseline correspondences between two key rigs come from. Our front end's patches
// do not match across minutes, so they come from descriptor matching outside VI-BA:
//   - SLAM Dunk's SIFT tracks (tools/viba_lc/lc_sift_matches.py in the main repository);
//   - LoMa matches of the other session (tools/viba_lc/lc_loma_matches.py);
//   - any other matcher that writes the same file, or another implementation of MatchSource.
//
// The file (CSV, one correspondence per line, header line first):
//   t_a_us,cam_a,t_b_us,cam_b,u_a,v_a,u_b,v_b[,score]
// t_a_us < t_b_us are image capture times in microseconds (the key rig times of VI-BA's input,
// within a tolerance), cam_a / cam_b VI-BA's camera indices, (u, v) pixels in VI-BA's convention
// (that of session_observations.csv).
namespace visual_inertial_ba::loop {

struct ImageMatches {
  int camA = 0, camB = 0;
  std::vector<Vec2> pxA, pxB;
};

class MatchSource {
 public:
  virtual ~MatchSource() = default;
  // capture times [us] of the images the source has correspondences for, sorted
  virtual std::vector<int64_t> imageTimesUs() const = 0;
  // earlier times with correspondences to the time tB (both sorted)
  virtual std::vector<int64_t> partnersBefore(int64_t tBUs) const = 0;
  // correspondences between the images at tA (earlier) and tB, one entry per camera combination
  virtual std::vector<ImageMatches> matches(int64_t tAUs, int64_t tBUs) const = 0;
};

// Loop tracks (tools/viba_lc/lc_sift_tracks.py): landmarks seen on several visits, one observation
// per line, header first:  track,visit,t_us,cam,u,v  (visit 0, 1, ... in time order)
struct LoopTrackObs {
  int64_t track = 0;
  int visit = 0;
  int64_t tUs = 0;
  int cam = 0;
  Vec2 px = Vec2::Zero();
};

class CsvMatchSource : public MatchSource {
 public:
  // returns null and sets *error if the file cannot be read
  static std::unique_ptr<CsvMatchSource> load(const std::string& path, std::string* error);

  // The correspondences between consecutive visits of the same loop track (see LoopTrackObs): for
  // visits v and v + 1, the pairs (observation of v, observation of v + 1) in time order, the first
  // maxPairsPerVisitPair of them (<= 0: all). The same as tools/viba_lc/lc_split_viba_input.py.
  static std::unique_ptr<CsvMatchSource> fromLoopTracks(
      const std::vector<LoopTrackObs>& obs,
      int maxPairsPerVisitPair);

  // (for tests and other producers)
  void add(int64_t tAUs, int camA, int64_t tBUs, int camB, const Vec2& pxA, const Vec2& pxB);
  size_t numCorrespondences() const {
    return numCorrespondences_;
  }

  std::vector<int64_t> imageTimesUs() const override;
  std::vector<int64_t> partnersBefore(int64_t tBUs) const override;
  std::vector<ImageMatches> matches(int64_t tAUs, int64_t tBUs) const override;

 private:
  // (tA, tB) -> (camA, camB) -> matches
  std::map<std::pair<int64_t, int64_t>, std::map<std::pair<int, int>, ImageMatches>> data_;
  size_t numCorrespondences_ = 0;
};

// returns false and sets *error if the file cannot be read
bool loadLoopTracks(const std::string& path, std::vector<LoopTrackObs>& out, std::string* error);

} // namespace visual_inertial_ba::loop
