#ifndef SEMANTIC_FUSIONPORTABLE_LABEL_H_
#define SEMANTIC_FUSIONPORTABLE_LABEL_H_

#include <cuda_runtime.h>
#include <string>
#include "nvblox/core/types.h"

namespace nvblox {
namespace semanticfusionportable {

/**
 * @brief A function to configure label of cityscapes/KITTI360
 * reference:
 * https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L60
 *
 * @param input_label
 * @param update_label
 * @return __host__
 */
__host__ __device__ inline void RemapSemanticFusionPortableLabel(
    const uint16_t& input_label, uint16_t* update_label) {
  switch (input_label) {
    case 0u:
      *update_label = 0u;
      // label_name = "Unlabeled";
      break;
    case 1u:
      *update_label = 1u;
      // label_name = "Road";
      break;
    case 2u:
      *update_label = 2u;
      // label_name = "Bike Path";
      break;
    case 3u:
      *update_label = 3u;
      // label_name = "Building";
      break;
    case 4u:
      *update_label = 4u;
      // label_name = "Wall";
      break;
    case 5u:
      *update_label = 5u;
      // label_name = "Fence";
      break;
    case 6u:
      *update_label = 6u;
      // label_name = "Pole";
      break;
    case 7u:
      *update_label = 7u;
      // label_name = "Traffic light";
      break;
    case 8u:
      *update_label = 8u;
      // label_name = "Traffic sign";
      break;
    case 9u:
      *update_label = 9u;
      // label_name = "Vegetation";
      break;
    case 10u:
      *update_label = 10u;
      // label_name = "Terrain";
      break;
    case 11u:
      *update_label = 11u;
      // label_name = "Sky";
      break;
    case 12u:
      *update_label = 12u;
      // label_name = "Person";
      break;
    case 13u:
      *update_label = 13u;
      // label_name = "Rider";
      break;
    case 14u:
      *update_label = 14u;
      // label_name = "Car";
      break;
    case 15u:
      *update_label = 15u;
      // label_name = "Truck";
      break;
    case 16u:
      *update_label = 16u;
      // label_name = "Bus";
      break;
    case 17u:
      *update_label = 17u;
      // label_name = "Train";
      break;
    case 18u:
      *update_label = 18u;
      // label_name = "Motorcycle";
      break;
    case 19u:
      *update_label = 19u;
      // label_name = "Bicycle";
      break;
    case 20u:
      *update_label = 20u;
      // label_name = "Curb";
      break;
    case 21u:
      *update_label = 21u;
      // label_name = "River";
      break;
    case 22u:
      *update_label = 22u;
      // label_name = "Road block";
      break;
    case 23u:
      *update_label = 23u;
      // label_name = "Sidewalk"
      break;
    default:
      *update_label = 0u;
      break;
  }
}

__host__ __device__ inline void updateLabelColorMap(const uint16_t& label,
                                                    Index3D* color) {
  switch (label) {
    case 0u:
      *color = Index3D(127, 127, 127);
      // label_name = "Unlabeled";
      break;
    case 1u:
      *color = Index3D(128, 64, 128);
      // label_name = "Road";
      break;
    case 2u:
      *color = Index3D(232, 35, 244);
      // label_name = "Bike Path";
      break;
    case 3u:
      *color = Index3D(70, 70, 70);
      // label_name = "Building";
      break;
    case 4u:
      *color = Index3D(156, 102, 102);
      // label_name = "Wall";
      break;
    case 5u:
      *color = Index3D(153, 153, 190);
      // label_name = "Fence";
      break;
    case 6u:
      *color = Index3D(153, 153, 153);
      // label_name = "Pole";
      break;
    case 7u:
      *color = Index3D(30, 170, 250);
      // label_name = "Traffic light";
      break;
    case 8u:
      *color = Index3D(0, 220, 220);
      // label_name = "Traffic sign";
      break;
    case 9u:
      *color = Index3D(35, 142, 107);
      // label_name = "Vegetation";
      break;
    case 10u:
      *color = Index3D(152, 251, 152);
      // label_name = "Terrain";
      break;
    case 11u:
      *color = Index3D(180, 130, 70);
      // label_name = "Sky";
      break;
    case 12u:
      *color = Index3D(60, 20, 220);
      // label_name = "Person";
      break;
    case 13u:
      *color = Index3D(0, 0, 255);
      // label_name = "Rider";
      break;
    case 14u:
      *color = Index3D(142, 0, 0);
      // label_name = "Car";
      break;
    case 15u:
      *color = Index3D(70, 0, 0);
      // label_name = "Truck";
      break;
    case 16u:
      *color = Index3D(100, 60, 0);
      // label_name = "Bus";
      break;
    case 17u:
      *color = Index3D(100, 80, 0);
      // label_name = "Train";
      break;
    case 18u:
      *color = Index3D(230, 0, 0);
      // label_name = "Motorcycle";
      break;
    case 19u:
      *color = Index3D(6, 109, 147);
      // label_name = "Bicycle";
      break;
    case 20:
      *color = Index3D(32, 11, 119);
      // label_name = "Curb";
      break;
    case 21u:
      *color = Index3D(152, 255, 244);
      // label_name = "River";
      break;
    case 22u:
      *color = Index3D(200, 178, 234);
      // label_name = "Road block";
      break;
    case 23u:
      *color = Index3D(232, 35, 244);
      // label_name = "Sidewalk"
      break;
    default:
      *color = Index3D(127, 127, 127);
      // label_name = "Unlabeled";
      break;
  }
}

}  // namespace semanticfusionportable
}  // namespace nvblox

#endif