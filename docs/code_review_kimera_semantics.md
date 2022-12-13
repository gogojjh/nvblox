# Kimera-Semantic

### Basic Data Structure
1. define semantic labels
```
// Consider id 0 to be the `unknown' label, for which we don't update the log-likelihood for that measurement.
  typedef vxb::AlignedVector<uint8_t> SemanticLabels;
  static constexpr uint8_t kUnknownSemanticLabelId = 0u;
// The size of this array determines how many semantic labels SemanticVoxblox supports.
// SemanticProbabilities[i] = prob: the probability of ith labels 
  static constexpr size_t kTotalNumberOfLabels = 21;
  typedef Eigen::Matrix<vxb::FloatingPoint, kTotalNumberOfLabels, 1>  SemanticProbabilities;
```

2. define semantic voxel
```
struct SemanticVoxel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Initialize voxel to unknown label.
  SemanticLabel semantic_label = 0u;
  // Initialize voxel to uniform probability.
  // Use log odds! So uniform ditribution of 1/kTotalNumberOfLabels,
  // should be std::log(1/kTotalNumberOfLabels)
  SemanticProbabilities semantic_priors =
      // SemanticProbabilities::Constant(std::log(1 / kTotalNumberOfLabels));
      SemanticProbabilities::Constant(-0.60205999132);
  // Initialize voxel with gray color
  // Make sure that all color maps agree on semantic label 0u -> gray
  HashableColor color = HashableColor::Gray();
};
```

3. color mode (semantic_integrator_base.cpp): ColorMode::kColor, ColorMode::kSemantic, ColorMode::kSemanticProbability
```
switch (semantic_config_.color_mode) {
  case ColorMode::kColor:
    // Nothing, base class colors the tsdf voxel for us.
    break;
  case ColorMode::kSemantic:
    tsdf_voxel->color = semantic_voxel->color;
    break;
  case ColorMode::kSemanticProbability:
    // TODO(Toni): Might be a bit expensive to calc all these exponentials...
    tsdf_voxel->color = vxb::rainbowColorMap(std::exp(semantic_voxel->semantic_priors[semantic_voxel->semantic_label]));
    break;
  default:
    LOG(FATAL) << "Unknown semantic color mode: "
                << static_cast<std::underlying_type<ColorMode>::type>(semantic_config_.color_mode);
    break;
}
```

### Code Pipeline
1. function to integrate semantic information
```
void FastSemanticTsdfIntegrator::integrateSemanticFunction(
    const vxb::Transformation& T_G_C,
    const vxb::Pointcloud& points_C,
    const vxb::Colors& colors,
    const SemanticLabels& semantic_labels,
    const bool freespace_points,
    vxb::ThreadSafeIndex* index_getter) {
  ///// create the ray_caster
  vxb::RayCaster ray_caster(origin, point_G, is_clearing, config_.voxel_carving_enabled, 
    config_.max_ray_length_m, voxel_size_inv_, 
    config_.default_truncation_distance, cast_from_origin);
  ///// update the value of the semantic voxel
  vxb::TsdfVoxel* voxel = allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);
  const float weight = getVoxelWeight(point_C);
  updateTsdfVoxel(origin, point_G, global_voxel_idx, color, weight, voxel);
  SemanticVoxel* semantic_voxel = allocateStorageAndGetSemanticVoxelPtr(global_voxel_idx, &semantic_block, &semantic_block_idx);
  SemanticProbabilities semantic_label_frequencies = SemanticProbabilities::Zero();
  semantic_label_frequencies[semantic_label] += 1.0f;
  updateSemanticVoxel(global_voxel_idx, semantic_label_frequencies, &mutexes_, voxel, semantic_voxel);  
}
'''

2. update semantic: update the probability of each label, get the label with the maximum prob, update the voxel_color
```
  ///// semantic_log_likelihood_ = log(1.0f - semantic_config_.semantic_measurement_probability = 0.8f) -> what is the equation?
  ///// [p1 p2 p3] += [p1_pri p2_pri p3_pri] + log_likehood * [frq_1 frq_2 frq_3]
  *semantic_prior_probability += semantic_log_likelihood_ * measurement_frequencies / measurement_count;
  ///// p_mle = max([p1 p2 p3])
  semantic_posterior.maxCoeff(semantic_label);
  ///// voxel_color = color(p_mle)
  *semantic_voxel_color = semantic_config_.semantic_label_to_color_->getColorFromSemanticLabel(semantic_label);
'''

### TODO
1. semantic information input
  * semantic information is obtained by the LiDAR
2. semantic integration and generate semantic mesh
3. semantic update
  * equation of the Bayesian update -> check the odd operation


