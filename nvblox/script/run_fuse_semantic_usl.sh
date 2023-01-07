# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/semanticusl_sequence12/ \
	--tsdf_integrator_max_integration_distance_m 100.0 \
	--tsdf_integrator_truncation_distance_vox 5.0 \
	--semantic_integrator_max_integration_distance_m 100.0 \
	--num_frames 300 \
	--voxel_size 0.25 \
	--mesh_integrator_min_weight 0.5 \
	--tsdf_frame_subsampling 1 \
	--mesh_frame_subsampling 20 \
	--esdf_frame_subsampling 10 \
	--semantic_frame_subsampling 1 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/semanticusl_sequence12_mesh_300.ply \
	# --esdf_output_path \
	# /Spy/dataset/mapping_results/nvblox/semanticusl_sequence12_esdf_test.ply \
	# --obstacle_output_path \
	# /Spy/dataset/mapping_results/nvblox/semanticusl_sequence12_obs_test.ply \
	# --esdf_mode 1 \
	# --esdf_zmin 0.5 \
	# --esdf_zmax 1.0 
