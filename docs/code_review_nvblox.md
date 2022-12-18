#### Code Pipeline of NVBlox

1. Data loader

2. Frame integration

	1. After loading data (the code API): Fuser::integrateFrame(const int frame_number)

	2. RgbdMapper::integrateOSLidarDepth -> ProjectiveTsdfIntegrator::integrateFrame -> ProjectiveTsdfIntegrator::integrateFrameTemplate 

		> * set <code>voxel_size</code> and <code>truncation_distance</code>, <code>truncation_distance_m = truncation_distance_vox * voxel_size</code>
		> * Identify blocks given the camera view: <code>view_calculator_.getBlocksInImageViewRaycast</code>
		>   * <code>getBlocksByRaycastingPixels</code>: Raycasts through (possibly subsampled) pixels in the image, use the kernal function
		>   * <code>combinedBlockIndicesInImageKernel</code>: retrieve visiable block by raycasting voxels (spacing carving), done in GPU
		> * TSDF integration given block indices: <code>integrateBlocksTemplate</code>
		>   * <code>ProjectiveTsdfIntegrator::integrateBlocks</code>: block integration for the OSLidar, use the kernal function
		>   * <code>integrateBlocksKernel</code>: TSDF integration for each block, done in GPU
		>     * <code>projectThreadVoxel</code>: convert blocks' indices into coordinates, retrieve voxels from the block, and project them onto the image to check whether they are visible or not
		>     * <code>interpolateOSLidarImage</code>: linear interpolation of depth images given float coordinates
		>       * ```const Index2D u_M_rounded = u_px.array().round().cast<int>();```
		>       * ```u_M_rounded.x() < 0 || u_M_rounded.y() < 0 || u_M_rounded.x() >= cols || u_M_rounded.y() >= rows)```: check bounds
		>     * <code>updateVoxel</code>: update the TSDF values of all visible voxels. 

3. Semantic frame integration
	1. ProjectiveSemanticIntegrator::integrateLidarFrame
	> * <code>block_indices = view_calculator_.getBlocksInImageViewRaycast()</code>
	>   * <code>AxisAlignedBoundingBox aabb_L = sensor.getViewAABB(T_L_C, 0.0f, max_integration_distance_m)</code>: get the AABB of the sensors' view. The maximum distance=max_integration_distance_m
	>		* <code>ViewCalculator::getBlocksByRaycastingPixels</code>: get valid blocks by raycasting each pixel along the ray in parallel (from the origin to depth + truncation_distance_m), get aabb_device_buffer[i] = true
	>   * <code>block_indices = reduceBlocksToThoseInTruncationBand</code>: remove blocks if they do not contain any voxels stay with the truncation band: 
	abs(voxel.distance) is smaller than truncation_distance_m

4. Weight averaging methods
    ```
    Projective distance:
        1: constant weight, truncate the fused_distance
        2: constant weight, truncate the voxel_distance_measured
        3: linear weight, truncate the voxel_distance_measured
        4: exponential weight, truncate the voxel_distance_measured
    Non-Projective distance:
        5: weight and distance derived from VoxField
        6: linear weight, distance derived from VoxField
    ```

3. Output data
    1. Mesh map
    2. ESDF map
    3. Obstacle map: points from the ESDF map whose distance is smaller than a threshold

4. Global planning test

### Detailed Pipeline of TSDF Integration 
1. integrateFrameTemplate: get visible block and allocate blocks of the layer 
-> integrateBlocksTemplate
    ```
    const std::vector<Index3D> block_indices =
        view_calculator_.getBlocksInImageViewRaycast(
            depth_frame, T_L_C, sensor, layer->block_size(),
            truncation_distance_m, max_integration_distance_m_);
    ```

2. integrateBlocksTemplate: copy block_indices/ptr (CPU) to block_indices/ptr (GPU)
-> integrateBlocks (each sensor)

3. integrateBlocks: get kVoxelsPerSide/ voxel_size/ truncation_distance_m
-> integrateBlocksKernel (each sensor)

4. integrateBlocksKernel (GPU): update each voxel within each block in parallel 
    ```
    integrateBlocksKernel
      <<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>()
    ```

### Detailed Pipeline of Semantic Integration 
1. integrateCameraFrame: get visible block and allocate blocks of the layer, generate depth image
-> integrateBlocksTemplate
    ```
    std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, color_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m);
    ...
    std::shared_ptr<const DepthImage> synthetic_depth_image_ptr =
      sphere_tracer_.renderImageOnGPU(
          camera, T_L_C, tsdf_layer, truncation_distance_m, MemoryType::kDevice,
          sphere_tracing_ray_subsampling_factor_);
    ```

2. integrateLidarFrame: get visible block and allocate blocks of the layer
-> integrateBlocksTemplate
    ```
    const std::vector<Index3D> block_indices =
      view_calculator_.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, sensor, layer->block_size(),
          truncation_distance_m, max_integration_distance_m_);
    ```

2. integrateBlocksTemplate: copy block_indices/ptr (CPU) to block_indices/ptr (GPU)
-> integrateBlocks (each sensor)
    
3. integrateBlocks: get kVoxelsPerSide/ voxel_size/ truncation_distance_m, the strategies for the camera and Lidar are different
-> integrateBlocksKernel (each sensor)

4. integrateBlocksKernel (GPU): update each voxel within each block in parallel 
    ```
    integrateBlocksKernel
      <<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>()
    ```


