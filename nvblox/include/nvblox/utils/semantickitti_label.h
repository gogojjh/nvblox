#ifndef SEMANTIC_KITTI_LABEL_H_
#define SEMANTIC_KITTI_LABEL_H_

#include <cuda_runtime.h>
#include <string>
#include "nvblox/core/types.h"

namespace nvblox {
namespace semantic_kitti {

/*
 * \brief A function to configure the original label of semantic KITTI
 * \reference:
 * https://github.com/VIS4ROB-lab/voxfield-panmap/blob/master/panoptic_mapping/include/panoptic_mapping/labels/semantic_kitti_all.yaml
// constexpr size_t kTotalNumberOfLabels = 26;
 */
__host__ __device__ inline void RemapSemanticKittiLabel(
    const uint16_t& input_label, uint16_t* update_label) {
  switch (input_label) {
    case 0u:
      *update_label = 0u;
      // label_name = "UnLabeled";
      break;
    case 1u:
      *update_label = 0u;
      break;
    case 10u:
      *update_label = 1u;
      // label_name = "CAR"
      break;
    case 11u:
      *update_label = 2u;
      // label_name = "BICYCLE"
      break;
    case 13u:
      *update_label = 3u;
      // label_name = "BUS"
      break;
    case 15u:
      *update_label = 4u;
      // label_name = "MOTORCYCLE"
      break;
    case 16u:
      *update_label = 5u;
      // label_name = "ON-RAILS"
      break;
    case 18u:
      *update_label = 6u;
      // label_name = "TRUCK"
      break;
    case 20u:
      *update_label = 7u;
      // label_name = "OTHER-VEHICLE"
      break;
    case 30u:
      *update_label = 8u;
      // label_name = "PERSON"
      break;
    case 31u:
      *update_label = 9u;
      // label_name = "BICYCLIST"
      break;
    case 32u:
      *update_label = 10u;
      // label_name = "MOTORCYCLIST"
      break;
    case 40u:
      *update_label = 11u;
      // label_name = "ROAD"
      break;
    case 44u:
      *update_label = 12u;
      // label_name = "PARKING"
      break;
    case 48u:
      *update_label = 13u;
      // label_name = "SIDEWALK"
      break;
    case 49u:
      *update_label = 14u;
      // label_name = "OTHER-GROUND"
      break;
    case 50u:
      *update_label = 15u;
      // label_name = "BUILDING"
      break;
    case 51u:
      *update_label = 16u;
      // label_name = "FENCE"
      break;
    case 52u:
      *update_label = 17u;
      // label_name = "OTHER-STRUCTURE"
      break;
    case 60u:
      *update_label = 18u;
      // label_name = "LANE-MARKING"
      break;
    case 70u:
      *update_label = 19u;
      // label_name = "VEGETATION"
      break;
    case 71u:
      *update_label = 20u;
      // label_name = "TRUNK"
      break;
    case 72u:
      *update_label = 21u;
      // label_name = "TERRAIN"
      break;
    case 80u:
      *update_label = 22u;
      // label_name = "POLE"
      break;
    case 81u:
      *update_label = 23u;
      // label_name = "TRAFFIC-SIGN"
      break;
    case 99u:
      *update_label = 24u;
      // label_name = "OTHER-OBJECT"
      break;
    case 252u:
      *update_label = 25u;
      // label_name = "MOVING-CAR"
      break;
    case 253u:
      *update_label = 26u;
      // label_name = "MOVING-BICYCLIST"
      break;
    case 254u:
      *update_label = 27u;
      // label_name = "MOVING-PERSON"
      break;
    case 255u:
      *update_label = 28u;
      // label_name = "MOVING-MOTORCYCLIST"
      break;
    case 256u:
      *update_label = 29u;
      // label_name = "MOVING-ON-RAILS"
      break;
    case 257u:
      *update_label = 30u;
      // label_name = "MOVING-BUS"
      break;
    case 258u:
      *update_label = 31u;
      // label_name = "MOVING-TRUCK"
      break;
    case 259u:
      *update_label = 32u;
      // label_name = "MOVING-OTHER-VEHICLE"
      break;
    default:
      *update_label = 0u;
      // label_name = "UNLABELED"
      break;
  }
}

// RGB
__host__ __device__ inline void updateLabelColorMap(const uint16_t& label,
                                                    Index3D* color) {
  switch (label) {
    case 0u:
      *color = Index3D(127, 127, 127);
      break;
    case 1u:
      *color = Index3D(100, 150, 245);
      break;
    case 2u:
      *color = Index3D(100, 230, 245);
      break;
    case 3u:
      *color = Index3D(100, 80, 250);
      break;
    case 4u:
      *color = Index3D(30, 60, 150);
      break;
    case 5u:
      *color = Index3D(0, 0, 255);
      break;
    case 6u:
      *color = Index3D(80, 30, 180);
      break;
    case 7u:
      *color = Index3D(0, 0, 255);
      break;
    case 8u:
      *color = Index3D(255, 30, 30);
      break;
    case 9u:
      *color = Index3D(255, 40, 200);
      break;
    case 10u:
      *color = Index3D(150, 30, 90);
      break;
    case 11u:
      *color = Index3D(255, 0, 255);
      break;
    case 12u:
      *color = Index3D(255, 150, 255);
      break;
    case 13u:
      *color = Index3D(75, 0, 75);
      break;
    case 14u:
      *color = Index3D(175, 0, 75);
      break;
    case 15u:
      *color = Index3D(255, 200, 0);
      break;
    case 16u:
      *color = Index3D(255, 120, 50);
      break;
    case 17u:
      *color = Index3D(255, 150, 0);
      break;
    case 18u:
      *color = Index3D(150, 255, 170);
      break;
    case 19u:
      *color = Index3D(0, 175, 0);
      break;
    case 20u:
      *color = Index3D(135, 60, 0);
      break;
    case 21u:
      *color = Index3D(150, 240, 80);
      break;
    case 22u:
      *color = Index3D(255, 240, 150);
      break;
    case 23u:
      *color = Index3D(255, 0, 0);
      break;
    case 24u:
      *color = Index3D(50, 255, 255);
      break;
    case 25u:
      *color = Index3D(100, 150, 245);
      break;
    case 26u:
      *color = Index3D(0, 0, 255);
      break;
    case 27u:
      *color = Index3D(255, 40, 200);
      break;
    case 28u:
      *color = Index3D(255, 30, 30);
      break;
    case 29u:
      *color = Index3D(150, 30, 90);
      break;
    case 30u:
      *color = Index3D(100, 80, 250);
      break;
    case 31u:
      *color = Index3D(80, 30, 180);
      break;
    case 32u:
      *color = Index3D(0, 0, 255);
      break;
    default:
      *color = Index3D(127, 127, 127);
      break;
  }
}

}  // namespace semantic_kitti
}  // namespace nvblox

#endif