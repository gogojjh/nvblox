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
#include <nvblox/integrators/projective_semantic_integrator.h>

#include "nvblox/core/color.h"
#include "nvblox/core/cuda/error_check.cuh"
#include "nvblox/core/interpolation_2d.h"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/utils/cityscapes_label.h"
#include "nvblox/utils/semanticfusionportable_label.h"
#include "nvblox/utils/semantickitti_label.h"
#include "nvblox/utils/timing.h"
#include "nvblox/utils/weight_function.h"

namespace nvblox {
template void ProjectiveSemanticIntegrator::integrateCameraFrame(
    const SemanticImage& semantic_frame, const Transform& T_L_C,
    const Camera& camera, const TsdfLayer& tsdf_layer,
    SemanticLayer* semantic_layer, std::vector<Index3D>* updated_blocks);

template void ProjectiveSemanticIntegrator::integrateCameraFrame(
    const SemanticImage& semantic_frame, const Transform& T_L_C,
    const CameraPinhole& camera, const TsdfLayer& tsdf_layer,
    SemanticLayer* semantic_layer, std::vector<Index3D>* updated_blocks);
}  // namespace nvblox

namespace nvblox {
__device__ inline bool updateSemanticVoxel(
    const int dataset_type,                                     // NOLINT
    const bool bayesian_semantics_enable,                       // NOLINT
    const uint16_t semantic_label,                              // NOLINT
    const SemanticLikelihoodFunction* semantic_log_likelihood,  // NOLINT
    SemanticVoxel* voxel_ptr) {
  // clang-format off
  uint16_t update_label;
  if (dataset_type == 1) {  // SemanticFusionPortable
    nvblox::semanticfusionportable::RemapSemanticFusionPortableLabel(semantic_label, &update_label);
    if (update_label == 0u) return false; // process unlabeled
  } else if (dataset_type == 3 || dataset_type == 5) {  // SemanticKitti, SemanticUSL
    nvblox::semantic_kitti::RemapSemanticKittiLabel(semantic_label, &update_label);
    if (update_label == 0u) return false;
  } else if (dataset_type == 6) {  // CityScapes
    nvblox::cityscapes::RemapCityScapesTrainLabel(semantic_label, &update_label);
    if (update_label == 0u) return false;
  } else if (dataset_type == 7) {  // CityScapes
    nvblox::cityscapes::RemapCityScapesLabel(semantic_label, &update_label);
    if (update_label == 0u) return false;
  }
  // clang-format on

  // updateSemanticVoxelProbabilities
  SemanticProbabilities measurement_frequency;
  measurement_frequency.setZero();
  // Label exceed the preset label range
  if (update_label >= measurement_frequency.size()) {
    return false;
  }
  measurement_frequency[update_label] += 1.0f;
  voxel_ptr->semantic_priors +=
      (*semantic_log_likelihood) * measurement_frequency;

  // ************************************************************************
  // TODO(gogojjh): handle the case if measurement_frequency is not binary
  // please refer to Kimera-Semantics (confusion matrix)
  // voxel_ptr->semantic_priors +=
  //     log((semantic_likelihood * measurement_frequency).normalized());
  // ************************************************************************

  // updateSemanticVoxel label by the MLE
  if (bayesian_semantics_enable) {
    voxel_ptr->semantic_priors.maxCoeff(&voxel_ptr->semantic_label);
  } else {
    // NOTE(gogojjh): not used Bayesian filter
    voxel_ptr->semantic_label = update_label;
  }
  return true;
}

__device__ inline bool interpolateLidarImage(
    const Lidar& lidar, const Vector3f& p_voxel_center_C, const float* image,
    const Vector2f& u_px, const int rows, const int cols,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    float* image_value) {
  // Try linear interpolation first
  interpolation::Interpolation2DNeighbours<float> neighbours;
  bool linear_interpolation_success = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, rows, cols, image_value, &neighbours);

  // Additional check
  // Check that we're not interpolating over a discontinuity
  // NOTE(alexmillane): This prevents smearing are object edges.
  if (linear_interpolation_success) {
    const float d00 = fabsf(neighbours.p00 - *image_value);
    const float d01 = fabsf(neighbours.p01 - *image_value);
    const float d10 = fabsf(neighbours.p10 - *image_value);
    const float d11 = fabsf(neighbours.p11 - *image_value);
    float maximum_depth_difference_to_neighbours =
        fmax(fmax(d00, d01), fmax(d10, d11));
    if (maximum_depth_difference_to_neighbours >
        linear_interpolation_max_allowable_difference_m) {
      linear_interpolation_success = false;
    }
  }

  // If linear didn't work - try nearest neighbour interpolation
  if (!linear_interpolation_success) {
    Index2D u_neighbour_px;
    if (!interpolation::interpolate2DClosest<
            float, interpolation::checkers::FloatPixelGreaterThanZero>(
            image, u_px, rows, cols, image_value, &u_neighbour_px)) {
      // If we can't successfully do closest, fail to intgrate this voxel.
      return false;
    }
    // Additional check
    // Check that this voxel is close to the ray passing through the pixel.
    // Note(alexmillane): This is to prevent large numbers of voxels
    // being integrated by a single pixel at long ranges.
    const Vector3f closest_ray = lidar.vectorFromPixelIndices(u_neighbour_px);
    const float off_ray_squared_distance =
        (p_voxel_center_C - p_voxel_center_C.dot(closest_ray) * closest_ray)
            .squaredNorm();
    if (off_ray_squared_distance >
        nearest_interpolation_max_allowable_squared_dist_to_ray_m) {
      return false;
    }
  }

  // TODO(alexmillane): We should add clearing rays, even in the case both
  // interpolations fail.
  return true;
}

// nearest_interpolation_max_allowable_squared_dist_to_ray_m, default: 0.125 **2
__device__ inline bool interpolateOSLidarImage(
    const OSLidar& lidar,                                            // NOLINT
    const Vector3f& p_voxel_center_C,                                // NOLINT
    const float* image,                                              // NOLINT
    const Vector2f& u_px,                                            // NOLINT
    const int rows,                                                  // NOLINT
    const int cols,                                                  // NOLINT
    const float linear_interp_max_allowable_difference_m,            // NOLINT
    const float nearest_interp_max_allowable_squared_dist_to_ray_m,  // NOLINT
    float* image_value) {
  // Try linear interpolation first
  interpolation::Interpolation2DNeighbours<float> neighbours;
  bool linear_interpolation_success = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, rows, cols, image_value, &neighbours);

  // Additional check
  // Check that we're not interpolating over a discontinuity
  // NOTE(alexmillane): This prevents smearing are object edges.
  if (linear_interpolation_success) {
    const float d00 = fabsf(neighbours.p00 - *image_value);
    const float d01 = fabsf(neighbours.p01 - *image_value);
    const float d10 = fabsf(neighbours.p10 - *image_value);
    const float d11 = fabsf(neighbours.p11 - *image_value);
    float maximum_depth_difference_to_neighbours =
        fmax(fmax(d00, d01), fmax(d10, d11));
    if (maximum_depth_difference_to_neighbours >
        linear_interp_max_allowable_difference_m) {
      linear_interpolation_success = false;
    }
  }

  // If linear didn't work - try nearest neighbour interpolation
  if (!linear_interpolation_success) {
    Index2D u_neighbour_px;
    if (!interpolation::interpolate2DClosest<
            float, interpolation::checkers::FloatPixelGreaterThanZero>(
            image, u_px, rows, cols, image_value, &u_neighbour_px)) {
      // If we can't successfully do closest, fail to intgrate this voxel.
      return false;
    }
    // Additional check
    // Check that this voxel is close to the ray passing through the pixel.
    // Note(alexmillane): This is to prevent large numbers of voxels
    // being integrated by a single pixel at long ranges.
    const Vector3f closest_ray = lidar.vectorFromPixelIndices(u_neighbour_px);
    const float off_ray_squared_distance =
        (p_voxel_center_C - p_voxel_center_C.dot(closest_ray) * closest_ray)
            .squaredNorm();
    if (off_ray_squared_distance >
        nearest_interp_max_allowable_squared_dist_to_ray_m) {
      return false;
    }
  }

  // TODO(alexmillane): We should add clearing rays, even in the case both
  // interpolations fail.
  return true;
}

__device__ inline bool getPointVectorOSLidar(const OSLidar& lidar,
                                             const Index2D& u_C, const int rows,
                                             const int cols,
                                             Vector3f& point_vector) {
  const float kFloatEpsilon = 1e-8;  // Used for weights
  if (u_C.x() < 0 || u_C.y() < 0 || u_C.x() >= cols || u_C.y() >= rows) {
    return false;
  } else {
    point_vector = lidar.unprojectFromImageIndex(u_C);
    if (point_vector.norm() < kFloatEpsilon) {
      return false;
    } else {
      return true;
    }
  }
}

__device__ inline bool getNormalVectorOSLidar(const OSLidar& lidar,
                                              const Index2D& u_C,
                                              const int rows, const int cols,
                                              Vector3f& normal_vector) {
  const float kFloatEpsilon = 1e-8;  // Used for weights
  if (u_C.x() < 0 || u_C.y() < 0 || u_C.x() >= cols || u_C.y() >= rows) {
    return false;
  } else {
    normal_vector = lidar.getNormalVector(u_C);
    if (normal_vector.norm() < kFloatEpsilon) {
      return false;
    } else {
      return true;
    }
  }
}

// **********************************************
// *********************** Camera
// **********************************************
template <typename CameraType>
__global__ void integrateCameraBlocksKernel(
    const Index3D* block_indices_device_ptr,                // NOLINT
    const CameraType camera,                                // NOLINT
    const uint16_t* semantic_image,                         // NOLINT
    const int semantic_rows,                                // NOLINT
    const int semantic_cols,                                // NOLINT
    const float* depth_image,                               // NOLINT
    const int depth_rows,                                   // NOLINT
    const int depth_cols,                                   // NOLINT
    const Transform T_C_L,                                  // NOLINT
    const float block_size,                                 // NOLINT
    const float truncation_distance_m,                      // NOLINT
    const float max_weight,                                 // NOLINT
    const float max_integration_distance,                   // NOLINT
    const int dataset_type,                                 // NOLINT
    const bool bayesian_semantics_enable,                   // NOLINT
    const int depth_subsample_factor,                       // NOLINT
    SemanticBlock** block_device_ptrs,                      // NOLINT
    SemanticLikelihoodFunction* semantic_log_likelihood) {  // NOLINT
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          &u_px, &voxel_depth_m, &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DLinear<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  // function 4: Get the closest semantic value
  // If we can't successfully do closest, fail to intgrate this voxel.
  uint16_t semantic_image_value;
  if (!interpolation::interpolate2DClosest<
          uint16_t, interpolation::checkers::PixelAlwaysValid<uint16_t>>(
          semantic_image, u_px, semantic_rows, semantic_cols,
          &semantic_image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  SemanticVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the semantic voxel (Camera)
  updateSemanticVoxel(dataset_type, bayesian_semantics_enable,
                      semantic_image_value, semantic_log_likelihood, voxel_ptr);
}

// **********************************************
// *********************** LiDAR
// **********************************************
__global__ void integrateLidarBlocksKernel(
    const Index3D* block_indices_device_ptr,                         // NOLINT
    const OSLidar lidar,                                             // NOLINT
    const uint16_t* semantic_image,                                  // NOLINT
    const float* depth_image,                                        // NOLINT
    const int rows, const int cols,                                  // NOLINT
    const Transform T_C_L,                                           // NOLINT
    const float block_size,                                          // NOLINT
    const float truncation_distance_m,                               // NOLINT
    const float max_weight,                                          // NOLINT
    const float max_integration_distance,                            // NOLINT
    const float linear_interp_max_allowable_difference_m,            // NOLINT
    const float nearest_interp_max_allowable_squared_dist_to_ray_m,  // NOLINT
    const int dataset_type,                                          // NOLINT
    const bool bayesian_semantics_enable,                            // NOLINT
    SemanticBlock** block_device_ptrs,                               // NOLINT
    SemanticLikelihoodFunction* semantic_log_likelihood) {           // NOLINT
  // function 1
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  //     - the projected image coordinate of the voxel
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, lidar, T_C_L, block_size,
                          &u_px, &voxel_depth_m, &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  // function 2: Interpolate on the depth_image plane
  float depth_image_value;
  if (!interpolateOSLidarImage(
          lidar, p_voxel_center_C, depth_image, u_px, rows, cols,
          linear_interp_max_allowable_difference_m,
          nearest_interp_max_allowable_squared_dist_to_ray_m,
          &depth_image_value)) {
    return;
  }

  // function 3: Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = depth_image_value - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  // function 4: Get the closest semantic value
  // If we can't successfully do closest, fail to intgrate this voxel.
  uint16_t semantic_image_value;
  if (!interpolation::interpolate2DClosest<
          uint16_t, interpolation::checkers::PixelAlwaysValid<uint16_t>>(
          semantic_image, u_px, rows, cols, &semantic_image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  SemanticVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the semantic voxel (LiDAR)
  updateSemanticVoxel(dataset_type, bayesian_semantics_enable,
                      semantic_image_value, semantic_log_likelihood, voxel_ptr);
}

// ***************************************************************
// ***************************************************************
// ***************************************************************
__global__ void updateColorBlocks(
    const int dataset_type,                            // NOLINT
    const SemanticBlock** block_device_ptrs_semantic,  // NOLINT
    ColorBlock** block_device_ptrs_color) {
  const SemanticVoxel* semantic_voxel_ptr =
      &(block_device_ptrs_semantic[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // clang-format off
  Index3D color;            // rgb
  if (dataset_type == 1) {  // SemanticFusionPortable
    nvblox::semanticfusionportable::updateLabelColorMap(uint16_t(semantic_voxel_ptr->semantic_label), &color);
  } else if (dataset_type == 3 || dataset_type == 5) {  // SemanticKitti, SemanticUSL
    nvblox::semantic_kitti::updateLabelColorMap(uint16_t(semantic_voxel_ptr->semantic_label), &color);
  } else if (dataset_type == 6 || dataset_type == 7) {  // CityScapes_Train, CityScapes
    nvblox::cityscapes::updateLabelColorMap(uint16_t(semantic_voxel_ptr->semantic_label), &color);
  }
  // clang-format on

  ColorVoxel* color_voxel_ptr =
      &(block_device_ptrs_color[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);
  color_voxel_ptr->color = Color(color.x(), color.y(), color.z());
}

ProjectiveSemanticIntegrator::ProjectiveSemanticIntegrator()
    : ProjectiveIntegratorBase() {
  // TODO(gogojh): initialize the semantic_log_likelihood function
  match_probability_ = 0.8f;
  non_match_probability_ = 0.2f;
  log_match_probability_ = std::log(match_probability_);
  log_non_match_probability_ = std::log(non_match_probability_);
  semantic_log_likelihood_ =
      semantic_log_likelihood_.setConstant(log_non_match_probability_);
  semantic_log_likelihood_.diagonal() =
      semantic_log_likelihood_.diagonal().Constant(log_match_probability_);

  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveSemanticIntegrator::~ProjectiveSemanticIntegrator() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveSemanticIntegrator::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

float ProjectiveSemanticIntegrator::
    lidar_linear_interpolation_max_allowable_difference_vox() const {
  return lidar_linear_interpolation_max_allowable_difference_vox_;
}

float ProjectiveSemanticIntegrator::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox() const {
  return lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_;
}

void ProjectiveSemanticIntegrator::
    lidar_linear_interpolation_max_allowable_difference_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_linear_interpolation_max_allowable_difference_vox_ = value;
}

void ProjectiveSemanticIntegrator::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = value;
}

int ProjectiveSemanticIntegrator::dataset_type() const { return dataset_type_; }

void ProjectiveSemanticIntegrator::dataset_type(int value) {
  CHECK_GE(value, 0);
  dataset_type_ = value;
}

bool ProjectiveSemanticIntegrator::bayesian_semantics_enabled() const {
  return bayesian_semantics_enabled_;
}

void ProjectiveSemanticIntegrator::bayesian_semantics_enabled(bool value) {
  CHECK_GE(value, 0);
  bayesian_semantics_enabled_ = value;
}

// *********************************************
// **************** Camera
// *********************************************
template <typename CameraType>
void ProjectiveSemanticIntegrator::integrateCameraFrame(
    const SemanticImage& semantic_frame, const Transform& T_L_C,
    const CameraType& camera, const TsdfLayer& tsdf_layer,
    SemanticLayer* semantic_layer, std::vector<Index3D>* updated_blocks) {
  timing::Timer tsdf_timer("semantic/integrate");
  CHECK_NOTNULL(semantic_layer);
  CHECK_EQ(tsdf_layer.block_size(), semantic_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      semantic_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Get visible blocks
  timing::Timer blocks_in_view_timer("semantic/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, semantic_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m);
  LOG(INFO) << "[semantic] retrieved block_indices size: "
            << block_indices.size();
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer(
      "semantic/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  // NOTE(gogojjh): comment to be removed
  LOG(INFO) << "[semantic] (remining after removal) block_indices size: "
            << block_indices.size();
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate semantic blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("semantic/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  allocate_blocks_timer.Stop();

  // Create a synthetic depth image
  timing::Timer sphere_trace_timer("semantic/integrate/sphere_trace");
  std::shared_ptr<const DepthImage> synthetic_depth_image_ptr =
      sphere_tracer_.renderImageOnGPU(
          camera, T_L_C, tsdf_layer, truncation_distance_m, MemoryType::kDevice,
          sphere_tracing_ray_subsampling_factor_);
  sphere_trace_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (GPU)
  timing::Timer update_blocks_timer("semantic/integrate/update_camera_blocks");
  integrateBlocksTemplate(block_indices, *synthetic_depth_image_ptr,
                          semantic_frame, T_L_C, camera, semantic_layer);
  const Transform T_C_L = T_L_C.inverse();
  integrateCameraBlocks(*synthetic_depth_image_ptr, semantic_frame, T_C_L,
                        camera, truncation_distance_m, semantic_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

template <typename CameraType>
void ProjectiveSemanticIntegrator::integrateCameraBlocks(
    const DepthImage& depth_frame, const SemanticImage& semantic_frame,
    const Transform& T_C_L, const CameraType& camera,
    const float& truncation_distance_m, SemanticLayer* layer_ptr) {
  // clang-format off
  CHECK_NOTNULL(layer_ptr);
  CHECK_EQ(semantic_frame.rows() % depth_frame.rows(), 0);
  CHECK_EQ(semantic_frame.cols() % depth_frame.cols(), 0);
  const int depth_subsampling_factor = semantic_frame.rows() / depth_frame.rows(); // default: 4
  CHECK_EQ(semantic_frame.cols() / depth_frame.cols(), depth_subsampling_factor);
  // clang-format on

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices_device_.size();

  // initialize some params related to semantic
  SemanticLikelihoodFunction* semantic_log_likelihood_device;
  cudaMalloc(&semantic_log_likelihood_device,
             sizeof(SemanticLikelihoodFunction));
  cudaMemcpy(semantic_log_likelihood_device, &semantic_log_likelihood_,
             sizeof(SemanticLikelihoodFunction), cudaMemcpyHostToDevice);

  integrateCameraBlocksKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                                integration_stream_>>>(
      block_indices_device_.data(),     // NOLINT
      camera,                           // NOLINT
      semantic_frame.dataConstPtr(),    // NOLINT
      semantic_frame.rows(),            // NOLINT
      semantic_frame.cols(),            // NOLINT
      depth_frame.dataConstPtr(),       // NOLINT
      depth_frame.rows(),               // NOLINT
      depth_frame.cols(),               // NOLINT
      T_C_L,                            // NOLINT
      layer_ptr->block_size(),          // NOLINT
      truncation_distance_m,            // NOLINT
      max_weight_,                      // NOLINT
      max_integration_distance_m_,      // NOLINT
      dataset_type_,                    // NOLINT
      bayesian_semantics_enabled_,      // NOLINT
      depth_subsampling_factor,         // NOLINT
      block_ptrs_device_.data(),        // NOLINT
      semantic_log_likelihood_device);  // NOLINT

  // Finish processing of the frame before returning control
  finish();
  cudaFree(semantic_log_likelihood_device);
  checkCudaErrors(cudaPeekAtLastError());
}

// *********************************************
// **************** LiDAR
// *********************************************
void ProjectiveSemanticIntegrator::integrateLidarFrame(
    const DepthImage& depth_frame, const SemanticImage& semantic_frame,
    const Transform& T_L_C, const OSLidar& lidar, const TsdfLayer& tsdf_layer,
    SemanticLayer* semantic_layer, std::vector<Index3D>* updated_blocks) {
  timing::Timer tsdf_timer("semantic/integrate");
  CHECK_NOTNULL(semantic_layer);
  CHECK_EQ(tsdf_layer.block_size(), semantic_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      semantic_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Identify blocks we can (potentially) see
  timing::Timer blocks_in_view_timer("semantic/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices =
      view_calculator_.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, lidar, semantic_layer->block_size(),
          truncation_distance_m, max_integration_distance_m_);
  blocks_in_view_timer.Stop();

  // ***********************************************************
  // NOTE(gogojjh): need to check the function
  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer(
      "semantic/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  // LOG_EVERY_N(INFO, 10)
  //     << "[semantic] (remining after removal) block_indices size: "
  //     << block_indices.size();
  blocks_in_band_timer.Stop();
  // ***********************************************************

  // Allocate blocks (CPU)
  timing::Timer allocate_blocks_timer("semantic/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  allocate_blocks_timer.Stop();

  // Update identified blocks
  timing::Timer update_blocks_timer("semantic/integrate/update_lidar_blocks");
  integrateBlocksTemplate(block_indices, depth_frame, semantic_frame, T_L_C,
                          lidar, semantic_layer);
  const Transform T_C_L = T_L_C.inverse();
  integrateLidarBlocks(depth_frame, semantic_frame, T_C_L, lidar, voxel_size,
                       truncation_distance_m, semantic_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

void ProjectiveSemanticIntegrator::integrateLidarBlocks(
    const DepthImage& depth_frame, const SemanticImage& semantic_frame,
    const Transform& T_C_L, const OSLidar& lidar, const float& voxel_size,
    const float& truncation_distance_m, SemanticLayer* layer_ptr) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices_device_.size();

  // Metric params
  const float linear_interpolation_max_allowable_difference_m =
      lidar_linear_interpolation_max_allowable_difference_vox_ * voxel_size;
  const float nearest_interpolation_max_allowable_squared_dist_to_ray_m =
      std::pow(lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ *
                   voxel_size,
               2);

  // initialize some params
  SemanticLikelihoodFunction* semantic_log_likelihood_device;
  cudaMalloc(&semantic_log_likelihood_device,
             sizeof(SemanticLikelihoodFunction));
  cudaMemcpy(semantic_log_likelihood_device, &semantic_log_likelihood_,
             sizeof(SemanticLikelihoodFunction), cudaMemcpyHostToDevice);

  integrateLidarBlocksKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                               integration_stream_>>>(
      block_indices_device_.data(),                               // NOLINT
      lidar,                                                      // NOLINT
      semantic_frame.dataConstPtr(),                              // NOLINT
      depth_frame.dataConstPtr(),                                 // NOLINT
      depth_frame.rows(),                                         // NOLINT
      depth_frame.cols(),                                         // NOLINT
      T_C_L,                                                      // NOLINT
      layer_ptr->block_size(),                                    // NOLINT
      truncation_distance_m,                                      // NOLINT
      max_weight_,                                                // NOLINT
      max_integration_distance_m_,                                // NOLINT
      linear_interpolation_max_allowable_difference_m,            // NOLINT
      nearest_interpolation_max_allowable_squared_dist_to_ray_m,  // NOLINT
      dataset_type_,                                              // NOLINT
      bayesian_semantics_enabled_,                                // NOLINT
      block_ptrs_device_.data(),                                  // NOLINT
      semantic_log_likelihood_device);                            // NOLINT

  // Finish processing of the frame before returning control
  finish();
  cudaFree(semantic_log_likelihood_device);
  checkCudaErrors(cudaPeekAtLastError());
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
template <typename SensorType>
void ProjectiveSemanticIntegrator::integrateBlocksTemplate(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const SemanticImage& semantic_frame, const Transform& T_L_C,
    const SensorType& sensor, SemanticLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);

  // Transfer to the device
  // NOTE(gogojjh): This is the key of transfering data from the host to the
  // device
  block_indices_device_ = block_indices_host_;
  block_ptrs_device_ = block_ptrs_host_;
}
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

// Synchronize color_layer using the semantic_layer
void ProjectiveSemanticIntegrator::updateColorLayer(
    const std::vector<Index3D>& block_indices,
    const SemanticLayer& semantic_layer, ColorLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();
  allocateBlocksWhereRequired(block_indices, layer_ptr);

  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;

  host_vector<const SemanticBlock*> block_ptrs_host_semantic;
  block_ptrs_host_semantic.reserve(num_blocks);
  host_vector<ColorBlock*> block_ptrs_host_color;
  block_ptrs_host_color.reserve(num_blocks);

  device_vector<const SemanticBlock*> block_ptrs_device_semantic;
  block_ptrs_device_semantic.reserve(num_blocks);
  device_vector<ColorBlock*> block_ptrs_device_color;
  block_ptrs_device_color.reserve(num_blocks);

  block_ptrs_host_semantic =
      getBlockPtrsFromIndices(block_indices, semantic_layer);
  block_ptrs_host_color = getBlockPtrsFromIndices(block_indices, layer_ptr);

  // Transfer to the device
  block_ptrs_device_semantic = block_ptrs_host_semantic;
  block_ptrs_device_color = block_ptrs_host_color;

  updateColorBlocks<<<num_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      dataset_type_,                      // NOLINT
      block_ptrs_device_semantic.data(),  // NOLINT
      block_ptrs_device_color.data());

  // Finish processing of the frame before returning control
  finish();
  checkCudaErrors(cudaPeekAtLastError());
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
__global__ void checkBlocksInTruncationBandSemantics(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr) {
  // A single thread in each block initializes the output to 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    contains_truncation_band_device_ptr[blockIdx.x] = 0;
  }
  // An function of CUDA to synchronize threads
  __syncthreads();

  // Get the Voxel we'll check in this thread
  const TsdfVoxel voxel = block_device_ptrs[blockIdx.x]
                              ->voxels[threadIdx.z][threadIdx.y][threadIdx.x];

  // If this voxel in the truncation band, write the flag to say that the block
  // should be processed.
  // NOTE(alexmillane): There will be collision on write here. However, from my
  // reading, all threads' writes will result in a single write to global
  // memory. Because we only write a single value (1) it doesn't matter which
  // thread "wins".
  if (std::abs(voxel.distance) <= truncation_distance_m && voxel.weight > 0) {
    contains_truncation_band_device_ptr[blockIdx.x] = true;
  }
}

std::vector<Index3D>
ProjectiveSemanticIntegrator::reduceBlocksToThoseInTruncationBand(
    const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Check 1) Are the blocks allocated
  // - performed on the CPU because the hash-map is on the CPU
  std::vector<Index3D> block_indices_check_1;
  block_indices_check_1.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tsdf_layer.isBlockAllocated(block_idx)) {
      block_indices_check_1.push_back(block_idx);
    }
  }

  if (block_indices_check_1.empty()) {
    return block_indices_check_1;
  }

  // Check 2) Does each of the blocks have a voxel within the truncation band
  // - performed on the GPU because the blocks are there
  // Get the blocks we need to check
  std::vector<const TsdfBlock*> block_ptrs =
      getBlockPtrsFromIndices(block_indices_check_1, tsdf_layer);

  const int num_blocks = block_ptrs.size();

  // Expand the buffers when needed
  if (num_blocks > truncation_band_block_ptrs_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserve(new_size);
    truncation_band_block_ptrs_device_.reserve(new_size);
    block_in_truncation_band_device_.reserve(new_size);
    block_in_truncation_band_host_.reserve(new_size);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_ = block_ptrs;
  truncation_band_block_ptrs_device_ = truncation_band_block_ptrs_host_;

  // Prepare output space
  block_in_truncation_band_device_.resize(num_blocks);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;

  checkBlocksInTruncationBandSemantics<<<num_thread_blocks, kThreadsPerBlock, 0,
                                         integration_stream_>>>(
      truncation_band_block_ptrs_device_.data(), truncation_distance_m,
      block_in_truncation_band_device_.data());

  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_ = block_in_truncation_band_device_;

  // Filter the indices using the result
  std::vector<Index3D> block_indices_check_2;
  block_indices_check_2.reserve(block_indices_check_1.size());
  for (int i = 0; i < block_indices_check_1.size(); i++) {
    if (block_in_truncation_band_host_[i] == true) {
      block_indices_check_2.push_back(block_indices_check_1[i]);
    }
  }

  return block_indices_check_2;
}

}  // namespace nvblox
