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
#pragma once

#include "nvblox/core/blox.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/lidar.h"
#include "nvblox/core/oslidar.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/projective_integrator_base.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/rays/sphere_tracer.h"

namespace nvblox {
/// A class performing TSDF intregration
///
/// Integrates a depth images into TSDF layers. The "projective" is a describes
/// one type of integration. Namely that voxels in view are projected into the
/// depth image (the alternative being casting rays out from the camera).
class ProjectiveSemanticIntegrator : public ProjectiveIntegratorBase {
 public:
  ProjectiveSemanticIntegrator();
  virtual ~ProjectiveSemanticIntegrator();

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the sensor. Supplied as a Transform mapping
  /// points in the sensor frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the semantic layer associated.
  /// Semantic integration is only performed on the voxels corresponding to the
  /// truncation band of this layer.
  /// @param semantic_layer A pointer to the layer into which this image will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  template <typename CameraType>
  void integrateCameraFrame(const SemanticImage& semantic_frame,
                            const Transform& T_L_C, const CameraType& camera,
                            const TsdfLayer& tsdf_layer,
                            SemanticLayer* semantic_layer,
                            std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param semantic_frame A semantic image.
  /// @param T_L_C The pose of the sensor. Supplied as a Transform mapping
  /// points in the sensor frame (C) to the layer frame (L).
  /// @param lidar A the lidar (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the semantic layer associated.
  /// Semantic integration is only performed on the voxels corresponding to the
  /// truncation band of this layer.
  /// @param semantic_layer A pointer to the layer into which this image will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateLidarFrame(const DepthImage& depth_frame,
                           const SemanticImage& semantic_frame,
                           const Transform& T_L_C, const OSLidar& lidar,
                           const TsdfLayer& tsdf_layer,
                           SemanticLayer* semantic_layer,
                           std::vector<Index3D>* updated_blocks = nullptr);

  /**
   * Update values of the color layer according to values of the semantic layer
   *
   * @param block_indices The blocks to be operated
   * @param semantic_layer The semantic layer
   * @param layer_ptr The color layer
   */
  void updateColorLayer(const std::vector<Index3D>& block_indices,
                        const SemanticLayer& semantic_layer,
                        ColorLayer* layer_ptr);

  /// Blocks until GPU operations are complete
  /// Ensure outstanding operations are finished (relevant for integrators
  /// launching asynchronous work)
  void finish() const override;

  /// A parameter getter
  /// The maximum allowable value for the maximum distance between the linearly
  /// interpolated image value and its four neighbours. Above this value we
  /// consider linear interpolation failed. This is to prevent interpolation
  /// across boundaries in the lidar image, which causing bleeding in the
  /// reconstructed 3D structure.
  /// @returns the maximum allowable distance in voxels
  float lidar_linear_interpolation_max_allowable_difference_vox() const;

  /// A parameter getter
  /// The maximum allowable distance between a reprojected voxel's center and
  /// the ray performing this integration. Above this we consider nearest
  /// nieghbour interpolation to have failed.
  /// @returns the maximum allowable distance in voxels
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox() const;

  /// A parameter setter
  /// see lidar_linear_interpolation_max_allowable_difference_vox()
  /// @param the new parameter value
  void lidar_linear_interpolation_max_allowable_difference_vox(float value);

  /// A parameter setter
  /// see lidar_nearest_interpolation_max_allowable_dist_to_ray_vox()
  /// @param the new parameter value
  void lidar_nearest_interpolation_max_allowable_dist_to_ray_vox(float value);

  int dataset_type() const;

  bool bayesian_semantics_enabled() const;

  /**
   * @brief Set the dataset type for parsing dataset label
   *
   * @param dataset_type
   */
  void dataset_type(int dataset_type);

  /**
   * @brief Set the bayesian semantics filter for updating dataset label
   *
   * @param bayesian_semantics_enabled
   */
  void bayesian_semantics_enabled(bool bayesian_semantics_enabled);

 protected:
  // Params
  // NOTE(alexmillane): See the getters above for a description.
  float lidar_linear_interpolation_max_allowable_difference_vox_ = 2.0f;
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = 0.5f;

  // Dataset type
  float dataset_type_ = 0;

  bool bayesian_semantics_enabled_ = true;

  // NOTE(gogojjh): Set the likelihood of semantic fusion
  float match_probability_, non_match_probability_;
  // Log probabilities of matching measurement and prior label, and
  // non-matching.
  float log_match_probability_, log_non_match_probability_;
  // confusion matrix:
  // A `#Labels X #Labels` Eigen matrix where each `j` column represents the
  // probability of observing label `j` when current label is `i`, where `i`
  // is the row index of the matrix.
  SemanticLikelihoodFunction semantic_log_likelihood_;

  /**
   * @brief Integrate the semantic_frame into each block
   *
   * @tparam CameraType
   * @param depth_frame
   * @param semantic_frame
   * @param T_C_L
   * @param camera
   * @param truncation_distance_m
   * @param layer_ptr
   */
  template <typename CameraType>
  void integrateCameraBlocks(const DepthImage& depth_frame,
                             const SemanticImage& semantic_frame,
                             const Transform& T_C_L, const CameraType& camera,
                             const float& truncation_distance_m,
                             SemanticLayer* layer_ptr);

  void integrateLidarBlocks(const DepthImage& depth_frame,
                            const SemanticImage& semantic_frame,
                            const Transform& T_C_L, const OSLidar& lidar,
                            const float& voxel_size,
                            const float& truncation_distance_m,
                            SemanticLayer* layer_ptr);

  template <typename SensorType>
  void integrateBlocksTemplate(const std::vector<Index3D>& block_indices,
                               const DepthImage& depth_frame,
                               const SemanticImage& semantic_frame,
                               const Transform& T_L_C, const SensorType& sensor,
                               SemanticLayer* layer_ptr);

  /**
   * @brief Takes a list of block indices and returns a subset containing the
   * block indices containing at least on voxel inside the truncation band of
   * the passed TSDF layer.
   *
   * @param block_indices
   * @param tsdf_layer
   * @param truncation_distance_m
   * @return std::vector<Index3D> Return the subset of block_indices containing
   * at least one voxel
   */
  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  // Params: the ratio to render the depth image
  int sphere_tracing_ray_subsampling_factor_ = 4;

  // Object to do ray tracing to generate occlusions
  SphereTracer sphere_tracer_;

  // Blocks to integrate on the current call (indices and pointers)
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<SemanticBlock*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<SemanticBlock*> block_ptrs_host_;

  // Buffers for getting blocks in truncation band
  device_vector<const TsdfBlock*> truncation_band_block_ptrs_device_;
  host_vector<const TsdfBlock*> truncation_band_block_ptrs_host_;
  device_vector<bool> block_in_truncation_band_device_;
  host_vector<bool> block_in_truncation_band_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox
