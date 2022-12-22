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

#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

#include "nvblox/core/color_map.h"

namespace nvblox {
namespace ColorMap {

__host__ device__ inline Color White() { return Color(255, 255, 255); }

__host__ device__ inline Color Black() { return Color(0, 0, 0); }

__host__ device__ inline Color Gray() { return Color(127, 127, 127); }

__host__ device__ inline Color Red() { return Color(255, 0, 0); }

__host__ device__ inline Color Green() { return Color(0, 255, 0); }

__host__ device__ inline Color Blue() { return Color(0, 0, 255); }

__host__ device__ inline Color Yellow() { return Color(255, 255, 0); }

__host__ device__ inline Color Orange() { return Color(255, 127, 0); }

__host__ device__ inline Color Purple() { return Color(127, 0, 255); }

__host__ device__ inline Color Teal() { return Color(0, 255, 255); }

__host__ device__ inline Color Pink() { return Color(255, 0, 127); }

__host__ __device__ inline Color rainbowColorMap(float h) {
  Color color;

  float s = 1.0;
  float v = 1.0;

  h -= floor(h);
  h *= 6;
  int i;
  float m, n, f;

  i = floor(h);
  f = h - i;
  if (!(i & 1)) f = 1 - f;  // if i is even
  m = v * (1 - s);
  n = v * (1 - s * f);

  switch (i) {
    case 6:
    case 0:
      color.r = 255 * v;
      color.g = 255 * n;
      color.b = 255 * m;
      break;
    case 1:
      color.r = 255 * n;
      color.g = 255 * v;
      color.b = 255 * m;
      break;
    case 2:
      color.r = 255 * m;
      color.g = 255 * v;
      color.b = 255 * n;
      break;
    case 3:
      color.r = 255 * m;
      color.g = 255 * n;
      color.b = 255 * v;
      break;
    case 4:
      color.r = 255 * n;
      color.g = 255 * m;
      color.b = 255 * v;
      break;
    case 5:
      color.r = 255 * v;
      color.g = 255 * m;
      color.b = 255 * n;
      break;
    default:
      color.r = 255;
      color.g = 127;
      color.b = 127;
      break;
  }
  return color;
}

}  // namespace ColorMap
}  // namespace nvblox
