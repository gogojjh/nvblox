# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07/ \
	--tsdf_integrator_max_integration_distance_m 100.0 \
	--tsdf_integrator_truncation_distance_vox 5.0 \
	--semantic_integrator_max_integration_distance_m 100.0 \
	--num_frames 1000 \
	--voxel_size 0.25 \
	--mesh_integrator_min_weight 0.5 \
	--tsdf_frame_subsampling 1 \
	--mesh_frame_subsampling 20 \
	--esdf_frame_subsampling 10 \
	--semantic_frame_subsampling 1 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_mesh_1000.ply \
	# --esdf_output_path \
	# /Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_nonground_esdf_1000.ply \
	# --obstacle_output_path \
	# /Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_nonground_obs_1000.ply \
	# --esdf_mode 1 \
	# --esdf_zmin 0.5 \
	# --esdf_zmax 1.0 
