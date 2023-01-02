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
__host__ __device__ inline void normalizeSemanticKittiLabel(
    const uint16_t& sem_kitti_label, uint16_t* update_label) {
  switch (sem_kitti_label) {
    case 0u:
      *update_label = 0u;
      break;
    case 1u:
      *update_label = 0u;
      break;
    case 10u:
      *update_label = 1u;
      // label_name = "Car";
      break;
    case 11u:
      *update_label = 2u;
      // label_name = "Bicycle";
      break;
    case 13u:
      *update_label = 5u;
      // label_name = "Bus";
      break;
    case 15u:
      *update_label = 3u;
      // label_name = "Motorcycle";
      break;
    case 16u:
      *update_label = 5u;
      // label_name = "Tram";
      break;
    case 18u:
      *update_label = 4u;
      // label_name = "Truck";
      break;
    case 20u:
      *update_label = 5u;
      // label_name = "OtherVehicle";
      break;
    case 30u:
      *update_label = 6u;
      // label_name = "Person";
      break;
    case 31u:
      *update_label = 7u;
      // label_name = "Bicyclist";
      break;
    case 32u:
      *update_label = 8u;
      // label_name = "Motorcyclist";
      break;
    case 40u:
      *update_label = 9u;
      // label_name = "Road";
      break;
    case 44u:
      *update_label = 10u;
      // label_name = "Parking";
      break;
    case 48u:
      *update_label = 11u;
      // label_name = "Sidewalk";
      break;
    case 49u:
      *update_label = 12u;
      // label_name = "Ground";
      break;
    case 50u:
      *update_label = 13u;
      // label_name = "Building";
      break;
    case 51u:
      *update_label = 15u;
      // label_name = "Fence";
      break;
    case 52u:
      *update_label = 16u;
      // label_name = "Structure";
      break;
    case 60u:
      *update_label = 9u;
      // label_name = "Roadmarking";
      break;
    case 70u:
      *update_label = 15u;
      // label_name = "Vegetation";
      break;
    case 71u:
      *update_label = 16u;
      // label_name = "Trunck";
      break;
    case 72u:
      *update_label = 17u;
      // label_name = "Terrain";
      break;
    case 80u:
      *update_label = 18u;
      // label_name = "Pole";
      break;
    case 81u:
      *update_label = 19u;
      // label_name = "Traffic_sign";
      break;
    case 99u:
      *update_label = 0u;
      // label_name = "Other_Object";
      break;
    case 252u:
      *update_label = 20u;
      // label_name = "Car";
      break;
    case 253u:
      *update_label = 21u;
      // label_name = "Bicycle";
      break;
    case 254u:
      *update_label = 22u;
      // label_name = "Person";
      break;
    case 255u:
      *update_label = 23u;
      // label_name = "Motorcycle";
      break;
    case 256u:
      *update_label = 24u;
      // label_name = "Tram";
      break;
    case 257u:
      *update_label = 24u;
      // label_name = "Bus";
      break;
    case 258u:
      *update_label = 25u;
      // label_name = "Truck";
      break;
    case 259u:
      *update_label = 24u;
      // label_name = "OtherVehicle";
      break;
    default:
      *update_label = 0u;
  }
}

__host__ __device__ inline void updateLabelColorMap(const uint16_t& label,
                                                    Index3D* color) {
  switch (label) {
    case 0u:
      *color = Index3D(127, 127, 127);
      break;
    case 1u:
      *color = Index3D(245, 150, 100);
      break;
    case 2u:
      *color = Index3D(245, 230, 100);
      break;
    case 5u:
      *color = Index3D(250, 80, 100);
      break;
    case 3u:
      *color = Index3D(150, 60, 30);
      break;
    case 4u:
      *color = Index3D(180, 30, 80);
      break;
    case 6u:
      *color = Index3D(30, 30, 255);
      break;
    case 7u:
      *color = Index3D(200, 40, 255);
      break;
    case 8u:
      *color = Index3D(90, 30, 150);
      break;
    case 9u:
      *color = Index3D(255, 0, 255);
      break;
    case 10u:
      *color = Index3D(255, 150, 255);
      break;
    case 11u:
      *color = Index3D(75, 0, 75);
      break;
    case 12u:
      *color = Index3D(75, 0, 175);
      break;
    case 13u:
      *color = Index3D(0, 200, 255);
      break;
    case 15u:
      *color = Index3D(50, 120, 255);
      break;
    case 16u:
      *color = Index3D(0, 150, 255);
      break;
    case 17u:
      *color = Index3D(80, 240, 150);
      break;
    case 18u:
      *color = Index3D(150, 240, 255);
      break;
    case 19u:
      *color = Index3D(0, 0, 255);
      break;
    case 20u:
      *color = Index3D(245, 150, 100);
      break;
    case 21u:
      *color = Index3D(255, 0, 0);
      break;
    case 22u:
      *color = Index3D(200, 40, 255);
      break;
    case 23u:
      *color = Index3D(30, 30, 255);
      break;
    case 24u:
      *color = Index3D(90, 30, 150);
      break;
    case 25u:
      *color = Index3D(180, 30, 80);
      break;
    default:
      *color = Index3D(127, 127, 127);
      break;
  }
}

}  // namespace semantic_kitti
}  // namespace nvblox
