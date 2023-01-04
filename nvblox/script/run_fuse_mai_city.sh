# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/mai_city_01/ \
	--tsdf_integrator_max_integration_distance_m 200.0 \
	--tsdf_integrator_truncation_distance_vox 3.0 \
	--semantic_integrator_max_integration_distance_m 70.0 \
	--num_frames 70 \
	--voxel_size 0.25 \
	--mesh_integrator_min_weight 0.5 \
	--tsdf_frame_subsampling 1 \
	--mesh_frame_subsampling 20 \
	--esdf_frame_subsampling -1 \
	--semantic_frame_subsampling -1 \
	--mesh_output_path /Spy/dataset/mapping_results/nvblox/mai_city_mesh_70.ply \
	# --esdf_output_path \
	# /Spy/dataset/mapping_results/nvblox/mai_city_esdf_test.ply \
	# --obstacle_output_path \
	# /Spy/dataset/mapping_results/nvblox/mai_city_obs_test.ply \
	# --esdf_mode 1 \
	# --esdf_zmin 0.5 \
	# --esdf_zmax 1.0 
