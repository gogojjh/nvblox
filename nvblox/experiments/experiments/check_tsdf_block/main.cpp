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
#include <fstream>
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/3dmatch.h"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/projective_semantic_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/utils/timing.h"

DECLARE_bool(alsologtostderr);

using namespace nvblox;

// Just a class so we can acces integrator internals
class ProjectiveTsdfIntegratorExperiment : public ProjectiveTsdfIntegrator {
 public:
  ProjectiveTsdfIntegratorExperiment() : ProjectiveTsdfIntegrator() {}
  virtual ~ProjectiveTsdfIntegratorExperiment(){};

  // Expose this publically
  template <typename SensorType>
  void integrateBlocksTemplate(const std::vector<Index3D>& block_indices,
                               const DepthImage& depth_frame,
                               const Transform& T_L_C, const SensorType& sensor,
                               const float truncation_distance_m,
                               TsdfLayer* layer) {
    ProjectiveTsdfIntegrator::integrateBlocksTemplate(
        block_indices, depth_frame, T_L_C, sensor, layer);
  }

  // Expose this publically
  //   std::vector<Index3D> getBlocksInViewUsingRaycasting(
  //       const DepthImage& depth_frame, const Transform& T_L_C,
  //       const Camera& camera, const float block_size) const {
  //     return ProjectiveTsdfIntegrator::getBlocksInViewUsingRaycasting(
  //         depth_frame, T_L_C, camera, block_size);
  //   }
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  const std::string dataset_base_path = "../../../tests/data/3dmatch";
  constexpr int kSeqNum = 1;

  constexpr float kVoxelSize = 0.05;
  TsdfLayer tsdf_layer(kVoxelSize, MemoryType::kDevice);

  ProjectiveTsdfIntegratorExperiment tsdf_integrator;
  ProjectiveSemanticIntegrator semantic_integrator;

  const unsigned int frustum_raycast_subsampling_rate = 4;
  tsdf_integrator.view_calculator().raycast_subsampling_factor(
      frustum_raycast_subsampling_rate);

  const float truncation_distance_m =
      tsdf_integrator.truncation_distance_vox() * kVoxelSize;

  // Update identified blocks (many times)
  constexpr int kNumIntegrations = 20;
  for (int i = 0; i < kNumIntegrations; i++) {
    // Load images
    auto image_loader_ptr =
        datasets::threedmatch::internal::createDepthImageLoader(
            dataset_base_path, kSeqNum);

    DepthImage depth_frame;
    CHECK(image_loader_ptr->getNextImage(&depth_frame));

    Eigen::Matrix3f camera_intrinsics;
    CHECK(datasets::threedmatch::internal::parseCameraFromFile(
        datasets::threedmatch::internal::getPathForCameraIntrinsics(
            dataset_base_path),
        &camera_intrinsics));
    const auto camera = Camera::fromIntrinsicsMatrix(
        camera_intrinsics, depth_frame.width(), depth_frame.height());

    Transform T_L_C;
    CHECK(datasets::threedmatch::internal::parsePoseFromFile(
        datasets::threedmatch::internal::getPathForFramePose(dataset_base_path,
                                                             kSeqNum, 0),
        &T_L_C));
    // std::cout << "T_L_C:" << std::endl << T_L_C.matrix() << std::endl;

    // Identify blocks we can (potentially) see (CPU)
    timing::Timer blocks_in_view_timer("tsdf/integrate/get_blocks_in_view");
    const std::vector<Index3D> block_indices =
        tsdf_integrator.view_calculator().getBlocksInImageViewRaycast(
            depth_frame, T_L_C, camera, tsdf_layer.block_size(),
            truncation_distance_m,
            tsdf_integrator.max_integration_distance_m());
    blocks_in_view_timer.Stop();
    LOG(INFO) << "Size of block_indices: " << block_indices.size();

    const std::vector<Index3D> block_indices_removal =
        semantic_integrator.reduceBlocksToThoseInTruncationBand(
            block_indices, tsdf_layer, truncation_distance_m);
    LOG(INFO) << "Size of block_indices (after removal): "
              << block_indices_removal.size();

    // Allocate blocks (CPU)
    timing::Timer allocate_blocks_timer("tsdf/integrate/allocate_blocks");
    allocateBlocksWhereRequired(block_indices, &tsdf_layer);
    allocate_blocks_timer.Stop();

    timing::Timer update_blocks_timer("tsdf/integrate/update_blocks");
    tsdf_integrator.integrateBlocksTemplate(block_indices, depth_frame, T_L_C,
                                            camera, truncation_distance_m,
                                            &tsdf_layer);
    update_blocks_timer.Stop();

    // Reset the layer such that we do TsdfBlock allocation.
    // tsdf_layer.clear();
  }

  std::cout << timing::Timing::Print() << std::endl;

  return 0;
}