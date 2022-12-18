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

 protected:
  // Params
  // NOTE(alexmillane): See the getters above for a description.
  float lidar_linear_interpolation_max_allowable_difference_vox_ = 2.0f;
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = 0.5f;

  template <typename SensorType>
  void updateBlocksTemplate(const std::vector<Index3D>& block_indices,
                            const DepthImage& depth_frame,
                            const SemanticImage& semantic_frame,
                            const Transform& T_L_C, const SensorType& sensor,
                            SemanticLayer* layer_ptr);

  void updateBlocks(const DepthImage& depth_frame,
                    const SemanticImage& semantic_frame, const Transform& T_C_L,
                    const OSLidar& lidar, SemanticLayer* layer_ptr);

  // Takes a list of block indices and returns a subset containing the block
  // indices containing at least on voxel inside the truncation band of the
  // passed TSDF layer.
  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

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
