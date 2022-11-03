/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
NOTE(jjiao):
This implements the operation that project OSLidar points onto a depth image
*/

#pragma once

#include "math.h"

#include <glog/logging.h>

namespace nvblox {

// default: 2048, 128, 45
OSLidar::OSLidar(int num_azimuth_divisions, int num_elevation_divisions,
                 float horizontal_fov_rad, float vertical_fov_rad)
    : num_azimuth_divisions_(num_azimuth_divisions),
      num_elevation_divisions_(num_elevation_divisions),
      horizontal_fov_rad_(horizontal_fov_rad / 180.0 * M_PI),
      vertical_fov_rad_(vertical_fov_rad / 180.0 * M_PI) {
  // Even numbers of beams allowed
  CHECK(num_azimuth_divisions_ % 2 == 0);

  // ****** TODO(jjiao): to be deleted_nvblox implementation
  // Angular distance between pixels
  // Note(alexmillane): Note the difference in division by N vs. (N-1) below.
  // This is because in the azimuth direction there's a wrapping around. The
  // point at pi/-pi is not double sampled, generating this difference.
  // ****************

  rads_per_pixel_elevation_ =
      vertical_fov_rad / static_cast<float>(num_elevation_divisions_ - 1);
  rads_per_pixel_azimuth_ =
      horizontal_fov_rad_ / static_cast<float>(num_azimuth_divisions_ - 1);

  // Inverse of the above
  elevation_pixels_per_rad_ = 1.0f / rads_per_pixel_elevation_;
  azimuth_pixels_per_rad_ = 1.0f / rads_per_pixel_azimuth_;

  // ****** TODO(jjiao): to be deleted_nvblox implementation
  // The angular lower-extremes of the image-plane
  // NOTE(alexmillane): Because beams pass through the angular extremes of the
  // FoV, the corresponding lower extreme pixels start half a pixel width
  // below this.
  // Note(alexmillane): Note that we use polar angle here, not elevation.
  // Polar is from the top of the sphere down, elevation, the middle up.
  // start_polar_angle_rad_ = M_PI / 2.0f - (vertical_fov_rad / 2.0f +
  //                                         rads_per_pixel_elevation_ / 2.0f);
  // start_azimuth_angle_rad_ = -M_PI - rads_per_pixel_azimuth_ / 2.0f;
  // *************

  // NOTE(jjiao): define the depth image
  // ********************* polar_angle
  // ****** the start polar_angle indicate the direction: x=0, -z
  // ****** the end polar_angle indicate the direction: x=0, +z
  // ********************* azimuth_angle
  // ****** the start azimuth_angle indicate the direction: x=0, +y
  // ****** the end azimuth_angle indicate the direction: x=0, -y
  // from bottom to top
  start_polar_angle_rad_ = -vertical_fov_rad / 2.0f;
  // from left to right
  start_azimuth_angle_rad_ = 0;
}

OSLidar::~OSLidar() {}

int OSLidar::num_azimuth_divisions() const { return num_azimuth_divisions_; }

int OSLidar::num_elevation_divisions() const {
  return num_elevation_divisions_;
}

float OSLidar::vertical_fov_rad() const { return vertical_fov_rad_; }

float OSLidar::horizontal_fov_rad() const { return horizontal_fov_rad_; }

int OSLidar::numel() const {
  return num_azimuth_divisions_ * num_elevation_divisions_;
}

int OSLidar::cols() const { return num_azimuth_divisions_; }

int OSLidar::rows() const { return num_elevation_divisions_; }

bool OSLidar::project(const Vector3f& p_C, Vector2f* u_C) const {
  const float rxy = p_C.head<2>().norm();
  constexpr float kMinProjectionEps = 0.01;
  if (p_C.norm() < kMinProjectionEps) {
    return false;
  }
  const float polar_angle_rad = atan2(p_C.z(), rxy);
  const float azimuth_angle_rad = M_PI - atan2(p_C.y(), p_C.x());

  // ****** TODO(jjiao): to be deleted_nvblox implementation
  // float v_float =
  //     (polar_angle_rad - start_polar_angle_rad_) * elevation_pixels_per_rad_;
  // float u_float =
  //     (azimuth_angle_rad - start_azimuth_angle_rad_) *
  //     azimuth_pixels_per_rad_;
  // ***********************

  // To image plane coordinates
  float v_float =
      (polar_angle_rad - start_polar_angle_rad_) / rads_per_pixel_elevation_;
  v_float = (num_elevation_divisions_ - 1) - v_float;
  float u_float =
      (azimuth_angle_rad - start_azimuth_angle_rad_) / rads_per_pixel_azimuth_;

  // Catch wrap around issues.
  if (u_float >= num_azimuth_divisions_) {
    u_float -= num_azimuth_divisions_;
  }

  // Points out of FOV
  // NOTE(alexmillane): It should be impossible to escape the -pi-to-pi range in
  // azimuth due to wrap around this. Therefore we don't check.
  if (v_float < -kMinProjectionEps ||
      v_float > num_elevation_divisions_ + kMinProjectionEps - 1) {
    return false;
  }

  // Write output
  *u_C = Vector2f(u_float, v_float);
  return true;
}

bool OSLidar::project(const Vector3f& p_C, Index2D* u_C) const {
  Vector2f u_C_float;
  bool res = project(p_C, &u_C_float);
  *u_C = u_C_float.array().round().matrix().cast<int>();
  return res;
}

float OSLidar::getDepth(const Vector3f& p_C) const { return p_C.norm(); }

Vector2f OSLidar::pixelIndexToImagePlaneCoordsOfCenter(
    const Index2D& u_C) const {
  // The index cast to a float is the coordinates of the lower corner of the
  // pixel.
  // return u_C.cast<float>() + Vector2f(0.5f, 0.5f);
  return u_C.cast<float>();
}

Index2D OSLidar::imagePlaneCoordsToPixelIndex(const Vector2f& u_C) const {
  // NOTE(alexmillane): We do round rather than a straight truncation such that
  // we handle negative image plane coordinates.
  return u_C.array().round().cast<int>();
}

// ***************************
Vector3f OSLidar::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                     const float depth) const {
  Vector3f p = depth * vectorFromImagePlaneCoordinates(u_C);
  // LOG(INFO) << "Project: " << u_C.y() << " " << u_C.x() << ", " << depth <<
  // "; "
  //           << p.x() << " " << p.y() << " " << p.z();
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f OSLidar::unprojectFromPixelIndices(const Index2D& u_C,
                                            const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

// ***************************
Vector3f OSLidar::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  float z = image::access<float>(round(u_C.y()), round(u_C.x()),
                                 num_azimuth_divisions_, z_image_ptr_cuda_);
  float depth =
      image::access<float>(round(u_C.y()), round(u_C.x()),
                           num_azimuth_divisions_, depth_image_ptr_cuda_);
  float r = sqrt(depth * depth - z * z);
  float azimuth_angle_rad = M_PI - u_C.x() * rads_per_pixel_azimuth_;

  // Vector3f p(-r * sin(azimuth_angle_rad), -z, r * cos(azimuth_angle_rad));
  // p /= depth;
  // printf(
  //     "ux: %f, uy: %f --- z: %f, depth: %f, r: %f, azimth: %f --- x: %f, y: "
  //     "%f, z: %f\n",
  //     u_C.x(), u_C.y(), z, depth, r, azimuth_angle_rad, p.x(), p.y(), p.z());

  return Vector3f(-r * sin(azimuth_angle_rad), -z, r * cos(azimuth_angle_rad)) /
         depth;
}

Vector3f OSLidar::vectorFromPixelIndices(const Index2D& u_C) const {
  return vectorFromImagePlaneCoordinates(
      pixelIndexToImagePlaneCoordsOfCenter(u_C));
}

AxisAlignedBoundingBox OSLidar::getViewAABB(const Transform& T_L_C,
                                            const float min_depth,
                                            const float max_depth) const {
  // The AABB is a square centered at the OSLidars location where the height is
  // determined by the OSLidar FoV.
  // NOTE(alexmillane): The min depth is ignored in this function, it is a
  // parameter so it matches with camera's getViewAABB()
  AxisAlignedBoundingBox box(
      Vector3f(-max_depth, -max_depth,
               -max_depth * sin(vertical_fov_rad_ / 2.0f)),
      Vector3f(max_depth, max_depth,
               max_depth * sin(vertical_fov_rad_ / 2.0f)));
  // LOG(INFO) << "max_depth: " << max_depth << ", " << vertical_fov_rad_ / 2.0
  //           << ", " << max_depth * sin(vertical_fov_rad_ / 2.0);
  // LOG(INFO) << "box_min: " << box.min().transpose()
  //           << ", box_max: " << box.max().transpose();

  // Translate the box to the sensor's location (note that orientation doesn't
  // matter as the OSLidar sees in the circle)
  box.translate(T_L_C.translation());
  // LOG(INFO) << "box_min: " << box.min().transpose()
  //           << ", box_max: " << box.max().transpose();
  return box;
}

size_t OSLidar::Hash::operator()(const OSLidar& OSLidar) const {
  // Taken from:
  // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
  size_t az_hash = std::hash<int>()(OSLidar.num_azimuth_divisions_);
  size_t el_hash = std::hash<int>()(OSLidar.num_elevation_divisions_);
  size_t fov_hash = std::hash<float>()(OSLidar.vertical_fov_rad_);
  return ((az_hash ^ (el_hash << 1)) >> 1) ^ (fov_hash << 1);
}

bool operator==(const OSLidar& lhs, const OSLidar& rhs) {
  return (lhs.num_azimuth_divisions_ == rhs.num_azimuth_divisions_) &&
         (lhs.num_elevation_divisions_ == rhs.num_elevation_divisions_) &&
         (std::fabs(lhs.vertical_fov_rad_ - rhs.vertical_fov_rad_) <
          std::numeric_limits<float>::epsilon());
}

}  // namespace nvblox
