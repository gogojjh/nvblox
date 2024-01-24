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
__host__ __device__ inline void RemapCityScapesTrainLabel(
    const uint16_t& input_label, uint16_t* update_label) {
  switch (input_label) {
    case 0u:
      *update_label = 7u;
      // label_name = "Road";
      break;
    case 1u:
      *update_label = 8u;
      // label_name = "Sidewalk";
      break;
    case 2u:
      *update_label = 11u;
      // label_name = "Building";
      break;
    case 3u:
      *update_label = 12u;
      // label_name = "Wall";
      break;
    case 4u:
      *update_label = 13u;
      // label_name = "Fence";
      break;
    case 5u:
      *update_label = 17u;
      // label_name = "Pole";
      break;
    case 6u:
      *update_label = 19u;
      // label_name = "Traffic light";
      break;
    case 7u:
      *update_label = 20u;
      // label_name = "Traffic sign";
      break;
    case 8u:
      *update_label = 21u;
      // label_name = "Vegetation";
      break;
    case 9u:
      *update_label = 22u;
      // label_name = "Terrain";
      break;
    case 10u:
      *update_label = 23u;
      // label_name = "Sky";
      break;
    case 11u:
      *update_label = 24u;
      // label_name = "Person";
      break;
    case 12u:
      *update_label = 25u;
      // label_name = "Rider";
      break;
    case 13u:
      *update_label = 26u;
      // label_name = "Car";
      break;
    case 14u:
      *update_label = 27u;
      // label_name = "Truck";
      break;
    case 15u:
      *update_label = 28u;
      // label_name = "Bus";
      break;
    case 16u:
      *update_label = 31u;
      // label_name = "Train";
      break;
    case 17u:
      *update_label = 32u;
      // label_name = "Motorcycle";
      break;
    case 18u:
      *update_label = 33u;
      // label_name = "Bicycle";
      break;
    default:
      *update_label = 0u;
      // label_name = "Unlabeled";
      break;
  }
}

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
  *update_label = input_label;
}

__host__ __device__ inline void updateLabelColorMap(const uint16_t& label,
                                                    Index3D* color) {
  switch (label) {
    case 0u: 
      *color = Index3D(  127, 127, 127);
      // label_name = 'Unlabeled';
      break;
    case 1u: 
      *color = Index3D(  127, 127, 127);
      // label_name = 'Ego vehicle';
      break;
    case 2u: 
      *color = Index3D(  127, 127, 127);
      // label_name = 'Rectification border';
      break;
    case 3u: 
      *color = Index3D(  127, 127, 127);
      // label_name = 'Out of roi';
      break;
    case 4u: 
      *color = Index3D(  127, 127, 127);
      // label_name = 'Static';
      break;
    case 5u: 
      *color = Index3D(111, 74,  0);
      // label_name = 'Dynamic';
      break;
    case 6u: 
      *color = Index3D( 81,  0, 81);
      // label_name = 'Ground';
      break;
    case 7u: 
      *color = Index3D(128, 64,128);
      // label_name = 'Road';
      break;
    case 8u: 
      *color = Index3D(244, 35,232);
      // label_name = 'Sidewalk';
      break;
    case 9u: 
      *color = Index3D(250, 170, 160);
      // label_name = 'Parking';
      break;
    case 10u: 
      *color = Index3D(230, 150, 140);
      // label_name = 'Rail track';
      break;
    case 11u: 
      *color = Index3D( 70, 70, 70);
      // label_name = 'Building'  ;
      break;
    case 12u: 
      *color = Index3D(102, 102, 156);
      // label_name = 'Wall'      ;
      break;
    case 13u: 
      *color = Index3D(190, 153, 153);
      // label_name = 'Fence'     ;
      break;
    case 14u: 
      *color = Index3D(180, 165, 180);
      // label_name = 'Guard rail';
      break;
    case 15u: 
      *color = Index3D(150, 100, 100);
      // label_name = 'Bridge'    ;
      break;
    case 16u: 
      *color = Index3D(150, 120, 90);
      // label_name = 'Tunnel'    ;
      break;
    case 17u: 
      *color = Index3D(153, 153, 153);
      // label_name = 'Pole'      ;
      break;
    case 18u: 
      *color = Index3D(153, 153, 153);
      // label_name = 'Polegroup' ;
      break;
    case 19u: 
      *color = Index3D(250, 170, 30);
      // label_name = 'Traffic light';
      break;
    case 20u: 
      *color = Index3D(220, 220,  0);
      // label_name = 'Traffic sign';
      break;
    case 21u: 
      *color = Index3D(107, 142, 35);
      // label_name = 'Vegetation';
      break;
    case 22u: 
      *color = Index3D(152, 251, 152);
      // label_name = 'Terrain'   ;
      break;
    case 23u: 
      *color = Index3D( 70, 130, 180);
      // label_name = 'Sky'       ;
      break;
    case 24u: 
      *color = Index3D(220, 20, 60);
      // label_name = 'Person'    ;
      break;
    case 25u: 
      *color = Index3D(255,  0,  0);
      // label_name = 'Rider'     ;
      break;
    case 26u: 
      *color = Index3D(  0,  0, 142);
      // label_name = 'Car'       ;
      break;
    case 27u: 
      *color = Index3D(  0,  0, 70);
      // label_name = 'Truck'     ;
      break;
    case 28u: 
      *color = Index3D(  0, 60, 100);
      // label_name = 'Bus'       ;
      break;
    case 29u: 
      *color = Index3D(  0,  0, 90);
      // label_name = 'Caravan'   ;
      break;
    case 30u: 
      *color = Index3D(  0,  0, 110);
      // label_name = 'Trailer'   ;
      break;
    case 31u: 
      *color = Index3D(  0, 80, 100);
      // label_name = 'Train'     ;
      break;
    case 32u: 
      *color = Index3D(  0,  0, 230);
      // label_name = 'Motorcycle';
      break;
    case 33u: 
      *color = Index3D(119, 11, 32);
      // label_name = 'Bicycle'   ;
      break;
  }
}

}  // namespace cityscapes
}  // namespace nvblox

#endif