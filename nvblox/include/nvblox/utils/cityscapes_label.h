#ifndef CITY_SCAPES_LABEL_H_
#define CITY_SCAPES_LABEL_H_

#include <cuda_runtime.h>
#include <string>
#include "nvblox/core/types.h"

namespace nvblox {
namespace cityscapes {

/**
 * @brief A function to configure label of cityscapes/KITTI360
 * reference:
 * https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L60
 *
 * @param input_label
 * @param update_label
 * @return __host__
 */
__host__ __device__ inline void RemapCityScapesLabel(
    const uint16_t& input_label, uint16_t* update_label) {
  switch (input_label) {
    case 0u:
      *update_label = 0u;
      // label_name = "Road";
      break;
    case 1u:
      *update_label = 1u;
      // label_name = "Sidewalk";
      break;
    case 2u:
      *update_label = 2u;
      // label_name = "Building";
      break;
    case 3u:
      *update_label = 3u;
      // label_name = "Wall";
      break;
    case 4u:
      *update_label = 4u;
      // label_name = "Fence";
      break;
    case 5u:
      *update_label = 5u;
      // label_name = "Pole";
      break;
    case 6u:
      *update_label = 6u;
      // label_name = "Traffic light";
      break;
    case 7u:
      *update_label = 7u;
      // label_name = "Traffic sign";
      break;
    case 8u:
      *update_label = 8u;
      // label_name = "Vegetation";
      break;
    case 9u:
      *update_label = 9u;
      // label_name = "Terrain";
      break;
    case 10u:
      *update_label = 10u;
      // label_name = "Sky";
      break;
    case 11u:
      *update_label = 11u;
      // label_name = "Person";
      break;
    case 12u:
      *update_label = 12u;
      // label_name = "Rider";
      break;
    case 13u:
      *update_label = 13u;
      // label_name = "Car";
      break;
    case 14u:
      *update_label = 14u;
      // label_name = "Truck";
      break;
    case 15u:
      *update_label = 15u;
      // label_name = "Bus";
      break;
    case 16u:
      *update_label = 16u;
      // label_name = "Train";
      break;
    case 17u:
      *update_label = 17u;
      // label_name = "Motorcycle";
      break;
    case 18u:
      *update_label = 18u;
      // label_name = "Bicycle";
      break;
    default:
      *update_label = 255u;
      break;
  }
}

__host__ __device__ inline void updateLabelColorMap(const uint16_t& label,
                                                    Index3D* color) {
  switch (label) {
    case 0:
      *color = Index3D(128, 64, 128);
      break;
    case 1:
      *color = Index3D(244, 35, 232);
      break;
    case 2:
      *color = Index3D(70, 70, 70);
      break;
    case 3:
      *color = Index3D(102, 102, 156);
      break;
    case 4:
      *color = Index3D(190, 153, 153);
      break;
    case 5:
      *color = Index3D(153, 153, 153);
      break;
    case 6:
      *color = Index3D(250, 170, 30);
      break;
    case 7:
      *color = Index3D(220, 220, 0);
      break;
    case 8:
      *color = Index3D(107, 142, 35);
      break;
    case 9:
      *color = Index3D(152, 251, 152);
      break;
    case 10:
      *color = Index3D(70, 130, 180);
      break;
    case 11:
      *color = Index3D(220, 20, 60);
      break;
    case 12:
      *color = Index3D(255, 0, 0);
      break;
    case 13:
      *color = Index3D(0, 0, 142);
      break;
    case 14:
      *color = Index3D(0, 0, 70);
      break;
    case 15:
      *color = Index3D(0, 60, 100);
      break;
    case 16:
      *color = Index3D(0, 80, 100);
      break;
    case 17:
      *color = Index3D(0, 0, 230);
      break;
    case 18:
      *color = Index3D(119, 11, 32);
      break;
    default:
      *color = Index3D(127, 127, 127);
      break;
  }
}

}  // namespace cityscapes
}  // namespace nvblox

#endif