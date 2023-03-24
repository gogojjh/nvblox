#### ProjectiveSemanticIntegrator

IntegrateCameraFrame
**Input:** ```semantic_frame, T_W_B, camera, tsdf_layer, semantic_layer, update_blocks```
```
  ///// integrateCameraFrame()
  block_indices = view_calculator_.getBlocksInViewPlanes();
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer);
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  synthetic_depth_image_ptr = sphere_tracer_.renderImageOnGPU();
  ///// integrateBlocksTemplate()
  block_indices_host_ = block_indices;  
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);
  ///// integrateCameraBlocksKernel()
  p_voxel_center_C, u_px = projectThreadVoxel();
  surface_depth_m = interpolate2DLinear_Depth();
  semantic_image_value = interpolate2DClosestSemantic(semantic_image);
  voxel_ptr = &(block_device_ptrs[blockIdx.x]->voxels[][][]);
  updateSemanticVoxel(voxel_ptr);
```

IntegrateLidarFrame
**Input:** ```depth_frame, semantic_frame, T_W_B, camera, tsdf_layer, semantic_layer, update_blocks```
```
  ///// integrateLidarFrame()
  block_indices = view_calculator_.getBlocksInViewPlanes();
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer);
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  ///// integrateBlocksTemplate()
  block_indices_host_ = block_indices;  
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);
  ///// integrateLidarBlocksKernel()
  p_voxel_center_C, u_px = projectThreadVoxel();
  surface_depth_m = interpolateOSLidarImage_Depth();
  semantic_image_value = interpolate2DClosestSemantic();
  voxel_ptr = &(block_device_ptrs[blockIdx.x]->voxels[][][]);
  updateSemanticVoxel(voxel_ptr);
```

