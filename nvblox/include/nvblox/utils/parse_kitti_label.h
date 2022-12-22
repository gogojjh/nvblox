#pragma once

#include <cuda_runtime.h>
#include <string>
#include "nvblox/core/types.h"

namespace nvblox {
namespace semantic_kitti {

constexpr size_t kTotalNumberOfLabels = 26;

/*
 * \brief A function to rearange the original label of semantic KITTI
 * \reference:
 * https://github.com/VIS4ROB-lab/voxfield-panmap/blob/master/panoptic_mapping/include/panoptic_mapping/labels/semantic_kitti_all.yaml
 */
__host__ __device__ inline void parseSemanticKittiLabel(
    const uint16_t& sem_kitti_label, uint16_t* update_label,
    Index3D* color_label) {
  switch (sem_kitti_label) {
    case 0u:
      *update_label = 0u;
      *color_label = Index3D(127, 127, 127);
      break;
    case 1u:
      *update_label = 0u;
      *color_label = Index3D(0, 0, 255);
      break;
    case 10u:
      *update_label = 1u;
      *color_label = Index3D(245, 150, 100);
      break;
    case 11u:
      *update_label = 2u;
      *color_label = Index3D(245, 230, 100);
      break;
    case 13u:
      *update_label = 5u;
      *color_label = Index3D(250, 80, 100);
      break;
    case 15u:
      *update_label = 3u;
      *color_label = Index3D(150, 60, 30);
      break;
    case 16u:
      *update_label = 5u;
      *color_label = Index3D(255, 0, 0);
      break;
    case 18u:
      *update_label = 4u;
      *color_label = Index3D(180, 30, 80);
      break;
    case 20u:
      *update_label = 6u;
      *color_label = Index3D(255, 0, 0);
      break;
    case 30u:
      *update_label = 6u;
      *color_label = Index3D(30, 30, 255);
      break;
    case 31u:
      *update_label = 7u;
      *color_label = Index3D(200, 40, 255);
      break;
    case 32u:
      *update_label = 8u;
      *color_label = Index3D(90, 30, 150);
      break;
    case 40u:
      *update_label = 9u;
      *color_label = Index3D(255, 0, 255);
      break;
    case 44u:
      *update_label = 10u;
      *color_label = Index3D(255, 150, 255);
      break;
    case 48u:
      *update_label = 11u;
      *color_label = Index3D(75, 0, 75);
      break;
    case 49u:
      *update_label = 12u;
      *color_label = Index3D(75, 0, 175);
      break;
    case 50u:
      *update_label = 13u;
      *color_label = Index3D(0, 200, 255);
      break;
    case 51u:
      *update_label = 15u;
      *color_label = Index3D(50, 120, 255);
      break;
    case 52u:
      *update_label = 16u;
      *color_label = Index3D(0, 150, 255);
      break;
    case 60u:
      *update_label = 9u;
      *color_label = Index3D(170, 255, 150);
      break;
    case 70u:
      *update_label = 15u;
      *color_label = Index3D(0, 175, 0);
      break;
    case 71u:
      *update_label = 16u;
      *color_label = Index3D(0, 60, 135);
      break;
    case 72u:
      *update_label = 17u;
      *color_label = Index3D(80, 240, 150);
      break;
    case 80u:
      *update_label = 18u;
      *color_label = Index3D(150, 240, 255);
      break;
    case 81u:
      *update_label = 19u;
      *color_label = Index3D(0, 0, 255);
      break;
    case 99u:
      *update_label = 0u;
      *color_label = Index3D(255, 255, 50);
      break;
    case 252u:
      *update_label = 20u;
      *color_label = Index3D(245, 150, 100);
      break;
    case 253u:
      *update_label = 21u;
      *color_label = Index3D(255, 0, 0);
      break;
    case 254u:
      *update_label = 22u;
      *color_label = Index3D(200, 40, 255);
      break;
    case 255u:
      *update_label = 23u;
      *color_label = Index3D(30, 30, 255);
      break;
    case 256u:
      *update_label = 24u;
      *color_label = Index3D(90, 30, 150);
      break;
    case 257u:
      *update_label = 24u;
      *color_label = Index3D(250, 80, 100);
      break;
    case 258u:
      *update_label = 25u;
      *color_label = Index3D(180, 30, 80);
      break;
    case 259u:
      *update_label = 24u;
      *color_label = Index3D(255, 0, 0);
      break;
    default:
      *update_label = 0u;
      *color_label = Index3D(127, 127, 127);
  }
}
}  // namespace semantic_kitti
}  // namespace nvblox
