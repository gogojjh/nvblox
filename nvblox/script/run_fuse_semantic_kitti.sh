# /bin/bash
cd build;
make && \
./executables/fuse_kitti \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07/ \
	--tsdf_integrator_max_integration_distance_m 70.0 \
	--semantic_integrator_max_integration_distance_m 70.0 \
	--num_frames 10 \
	--voxel_size 0.1 \
	--tsdf_frame_subsampling 1 \
	--mesh_frame_subsampling 20 \
	--esdf_frame_subsampling 10 \
	--semantic_frame_subsampling 1 \
	--esdf_mode 1 \
	--esdf_zmin 0.5 \
	--esdf_zmax 1.0 \
	--mesh_output_path \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_mesh_test.ply \
	--esdf_output_path \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_esdf_test.ply \
	--obstacle_output_path \
	/Spy/dataset/mapping_results/nvblox/semantickitti_sequence07_obs_test.ply \
