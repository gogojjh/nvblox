<h4>Semantic Integration</h4>

Integrated TSDF block: 19766               
[semantic] retrieved block_indices size: 3134
[semantic] (remining after removal) block_indices size: 577

<h4>Color Integration</h4>

Integrated TSDF block: 19766               
[color] retrieved block_indices size: 3134
[color] (remining after removal) block_indices size: 577

<code>printf("%f %f %u; ", u_px.x(), u_px.y(), semantic_image_value);</code>
for unsigned short <-> uint16_t



### Question

1. check Pose: T_W_sensor 

**semantics integration:**

  0.164075 -0.0920579   0.982143    3.51714
  0.986417 0.00743191  -0.164093  -0.320149
0.00780682   0.995726  0.0920268  -0.653417
         0          0          0          1

**color integration:**

  0.204409 -0.0992149   0.973845    4.35533
   0.97885  0.0121828  -0.204218  -0.505297
0.00839731   0.994991  0.0996068  -0.643031
         0          0          0          1
