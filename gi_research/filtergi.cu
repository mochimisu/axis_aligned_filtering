/*
* arealight.cu
* Area Light Filtering
* Adapted from NVIDIA OptiX Tutorial
* Brandon Wang, Soham Mehta
*/

#include "filtergi.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

rtDeclareVariable(float,          light_sigma, , );

// Create ONB from normal.  Resulting W is Parallel to normal
__device__ __inline__ void createONB( const optix::float3& n,
  optix::float3& U,
  optix::float3& V,
  optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( fabsf( U.x) < 0.001f && fabsf( U.y ) < 0.001f && fabsf( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalalized vector
__device__ __inline__ void createONB( const optix::float3& n,
  optix::float3& U,
  optix::float3& V)
{
  using namespace optix;
  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );
  if ( dot(U, U) < 1.e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );
  U = normalize( U );
  V = cross( n, U );
}

// sample hemisphere with cosine density
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
  const optix::float3& U,
  const optix::float3& V,
  const optix::float3& W,
  optix::float3& point )
{
  using namespace optix;

  float phi = 2.0f * M_PIf*sample.x;
  float r = sqrt( sample.y );
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x*x -y*y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  point = x*U + y*V + z*W;
}

// HeatMap visualization
__device__ __inline__ float3 heatMap(float val) {
  float fraction;
  if (val < 0.0f)
    fraction = -1.0f;
  else if (val > 1.0f)
    fraction = 1.0f;
  else
    fraction = 2.0f * val - 1.0f;

  if (fraction < -0.5f)
    return make_float3(0.0f, 2*(fraction+1.0f), 1.0f);
  else if (fraction < 0.0f)
    return make_float3(0.0f, 1.0f, 1.0f - 2.0f * (fraction + 0.5f));
  else if (fraction < 0.5f)
    return make_float3(2.0f * fraction, 1.0f, 0.0f);
  else
    return make_float3(1.0f, 1.0f - 2.0f*(fraction - 0.5f), 0.0f);
}

rtBuffer<float, 1>              gaussian_lookup;


// Our Gaussian Filter, based on w_xf
__device__ __inline__ float gaussFilter(float distsq, float wxf)
{
  float sample = distsq*wxf*wxf;
  if (sample > 0.9999) {
    return 0.0;
  }

  return exp(-3*sample);
}


//marsaglia polar method
__device__ __inline__ float2 randomGauss(float center, float std_dev, float2 sample)
{
  float u,v,s;
  u = sample.x * 2 - 1;
  v = sample.y * 2 - 1;
  s = u*u + v*v;
  float2 result = make_float2(
    center+std_dev*v*sqrt(-2.0*log(s)/s),
    center+std_dev*u*sqrt(-2.0*log(s)/s));
  return result;
}

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

rtDeclareVariable(float3, bg_color, , );

rtBuffer<float3, 2>               brdf;
//divided occlusion, undivided occlusion, wxf, num_samples
rtBuffer<float3, 2>               vis;
rtBuffer<float, 2>                vis_blur1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float2, 2>               slope_filter1d;
rtBuffer<float, 2>                spp_filter1d;
rtBuffer<float, 2>                spp;
rtBuffer<float, 2>                spp_cur;

//s1,s2
rtBuffer<float2, 2>               slope;
rtBuffer<uint, 2>                 use_filter_n;
rtBuffer<uint, 2>                 use_filter_occ;
rtBuffer<uint, 2>                 use_filter_occ_filter1d;
rtBuffer<float, 2>                proj_d;
rtBuffer<int, 2>                  obj_id_b;
rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           blur_occ, , );
rtDeclareVariable(uint,           blur_wxf, , );
rtDeclareVariable(uint,           err_vis, , );
rtDeclareVariable(uint,           view_mode, , );

rtDeclareVariable(uint,           normal_rpp, , );
rtDeclareVariable(uint,           brute_rpp, , );
rtDeclareVariable(uint,           max_rpp_pass, , );
rtDeclareVariable(uint,           show_progressive, , );
rtDeclareVariable(float,          zmin_rpp_scale, , );
rtDeclareVariable(int2,           pixel_radius, , );
rtDeclareVariable(int2,           pixel_radius_wxf, , );

rtDeclareVariable(uint,           show_brdf, , );
rtDeclareVariable(uint,           show_occ, , );

rtDeclareVariable(float,          max_disp_val, , );
rtDeclareVariable(float,          min_disp_val, , );

rtDeclareVariable(float,          spp_mu, , );

// Compute SPP given s1,s2,wxf @ a given pixel
__device__ __inline__ float computeSpp( float s1, float s2, float wxf ) {
  float spp_t_1 = (1/(1+s2)+proj_d[launch_index]*wxf);
  float spp_t_2 = (1+light_sigma * min(s1*wxf,1/proj_d[launch_index] * s1/(1+s1)));
  float spp = 4*spp_t_1*spp_t_1*spp_t_2*spp_t_2;
  return spp;
}

// Compute wxf given s2 @ a given pixel
__device__ __inline__ float computeWxf( float s2 ) {
  return min(spp_mu/(light_sigma * s2), 1/(proj_d[launch_index]*(1+s2)));
}

RT_PROGRAM void pinhole_camera_initial_sample() {
  // Find direction to shoot ray in
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  // Initialize the stuff we use in later passes
  vis[launch_index] = make_float3(1.0, 0.0, 0.0);
  slope[launch_index] = make_float2(0.0, 10000.0);
  float current_spp = normal_rpp * normal_rpp;
  use_filter_n[launch_index] = false;
  use_filter_occ[launch_index] = false;
  use_filter_occ_filter1d[launch_index] = false;
  brdf[launch_index] = make_float3(0,0,0);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);
  prd.first_pass = true;
  prd.sqrt_num_samples = normal_rpp;
  prd.unavg_vis = 0.0f;
  prd.vis_weight_tot = 0.0f;
  prd.hit_shadow = false;
  prd.use_filter_n = false;
  prd.s1 = slope[launch_index].x;
  prd.s2 = slope[launch_index].y;
  prd.hit = false;
  prd.obj_id = -1;

  rtTrace(top_object, ray, prd);

  obj_id_b[launch_index] = prd.obj_id;

  if (!prd.hit) {
    vis[launch_index] = make_float3(1,1,0);
    spp[launch_index] = 0;
    spp_cur[launch_index] = 0;
    brdf[launch_index].x = -2;
    return;
  }

  //Currently assuming fov of 60deg, height of 720p, 1:1 aspect
  float proj_dist = 1.0/360.0 * (prd.t_hit*tan(30.0*M_PI/180.0));
  proj_d[launch_index] = proj_dist;
  float wxf = computeWxf(prd.s2);
  float theoretical_spp = 0;
  if(prd.hit_shadow)
    theoretical_spp = computeSpp(prd.s1, prd.s2, wxf);

  world_loc[launch_index] = prd.world_loc;
  brdf[launch_index] = prd.brdf;
  n[launch_index] = normalize(prd.n);

  slope[launch_index] = make_float2(prd.s1, prd.s2);
  use_filter_n[launch_index] = prd.use_filter_n;
  use_filter_occ[launch_index] = prd.hit_shadow;
  vis[launch_index].x = 1;

  spp_cur[launch_index] = current_spp;
  spp[launch_index] = min(theoretical_spp, (float) brute_rpp * brute_rpp);
  spp_filter1d[launch_index] = spp[launch_index];

  if (prd.hit_shadow && prd.vis_weight_tot > 0.01) {
    vis[launch_index].x = prd.unavg_vis/prd.vis_weight_tot;
  }
  vis[launch_index].y = prd.unavg_vis;
  vis[launch_index].z = prd.vis_weight_tot;
}

RT_PROGRAM void pinhole_camera_continue_sample() {
  if (brdf[launch_index].x < -1.0f)
    return;
  float2 cur_slope = slope[launch_index];
  float wxf = computeWxf(cur_slope.y);
  float target_spp = computeSpp(cur_slope.x, cur_slope.y, wxf);
  spp[launch_index] = target_spp;
  float cur_spp = spp_cur[launch_index];

  // Compute spp difference
  if (cur_spp < target_spp ) {
    size_t2 screen = output_buffer.size();

    float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
    PerRayData_radiance prd;

    prd.first_pass = false;
    prd.unavg_vis = vis[launch_index].y;
    prd.vis_weight_tot = vis[launch_index].z;
    prd.hit_shadow = false;
    prd.use_filter_n = use_filter_n[launch_index];
    prd.s1 = slope[launch_index].x;
    prd.s2 = slope[launch_index].y;

    optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon); 
    int new_samp = min((int) (target_spp - cur_spp), (int) max_rpp_pass*max_rpp_pass);
    int sqrt_samp = ceil(sqrt((float)new_samp));
    prd.sqrt_num_samples = sqrt_samp;
    cur_spp = cur_spp + sqrt_samp * sqrt_samp;

    spp_cur[launch_index] = cur_spp;

    rtTrace(top_object, ray, prd);
    if (!prd.hit)
      return;
    vis[launch_index].z = prd.vis_weight_tot;
    vis[launch_index].y = prd.unavg_vis;
    if (prd.hit_shadow && prd.vis_weight_tot > 0.01) {
      vis[launch_index].x = prd.unavg_vis/prd.vis_weight_tot;
    }
  }

}


RT_PROGRAM void display_camera() {
  float3 cur_vis = vis[launch_index];
  float blurred_vis = cur_vis.x;
  float wxf = computeWxf(slope[launch_index].y);

  if (brdf[launch_index].x < -1.0f) {
    output_buffer[launch_index] = make_color( bg_color );
    return;
  }

  float3 brdf_term = make_float3(1);
  float vis_term = 1;
  if (show_brdf)
    brdf_term = brdf[launch_index];
  if (show_occ)
    vis_term = blurred_vis;
  if (view_mode) {
    if (view_mode == 1)
      //Occlusion only
      output_buffer[launch_index] = make_color( make_float3(blurred_vis) );
    if (view_mode == 2)  {
      //Scale
      //output_buffer[launch_index] = make_color( make_float3(scale) );
      float min_wxf = computeWxf(min_disp_val);
      float vis_color = 1/(wxf*light_sigma) * 8.0;
      output_buffer[launch_index] = make_color( heatMap(vis_color) );
      if (vis_color > 5)
        output_buffer[launch_index] = make_color( make_float3(0) );
    }
    if (view_mode == 3) 
      //Current SPP
      //output_buffer[launch_index] = make_color( make_float3(spp_cur[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp_cur[launch_index] / 60.0 ) );
    if (view_mode == 4) 
      //Theoretical SPP
      //output_buffer[launch_index] = make_color( make_float3(spp[launch_index]) / 100.0 );
      output_buffer[launch_index] = make_color( heatMap(spp[launch_index] / 60.0 ) );
    if (view_mode == 5)
      //Use filter (normals)
      output_buffer[launch_index] = make_color( make_float3(use_filter_n[launch_index])  );
    if (view_mode == 6)
      //Use filter (unocc)
      output_buffer[launch_index] = make_color( make_float3(use_filter_occ[launch_index])  );
    if (view_mode == 7)
      //View areas that are not yet converged to theoretical spp
      output_buffer[launch_index] = make_color( make_float3(spp_cur[launch_index] < spp[launch_index],0,0) );
    if (view_mode == 8)
      output_buffer[launch_index] = make_color( heatMap( (float)(obj_id_b[launch_index]-10)/5.0 ) );
  } else
    output_buffer[launch_index] = make_color( vis_term * brdf_term);

}

rtBuffer<BasicLight>        lights;
rtDeclareVariable(float3,        lightnorm, , );

__device__ __inline__ void occlusionFilter( float& blurred_vis_sum,
  float& sum_weight, const optix::float3& cur_world_loc, float3 cur_n,
  float wxf, int i, int j, const optix::size_t2& buf_size, 
  unsigned int pass ) {
    const float dist_scale_threshold = 10.0f;
    const float dist_threshold = 1.0f;
    const float angle_threshold = 20.0f * M_PI/180.0f;

    if (i > 0 && i < buf_size.x && j > 0 && j <buf_size.y) {
      uint2 target_index = make_uint2(i,j);
      float3 target_vis = vis[target_index];
      float target_wxf = computeWxf(slope[target_index].y);
      if (target_wxf > 0 && abs(wxf - target_wxf) < dist_scale_threshold &&
        use_filter_n[target_index]) {
          float3 target_loc = world_loc[target_index];
          float3 diff = cur_world_loc - target_loc;
          float euclidean_distancesq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
          float normcomp = dot(diff, lightnorm);
          float distancesq = euclidean_distancesq - normcomp*normcomp;
          if (distancesq < 10.0) {
            float3 target_n = n[target_index];
            if (acos(dot(target_n, cur_n)) < angle_threshold) {
              float weight = gaussFilter(distancesq, wxf);
              float target_vis_val = target_vis.x;
              if (pass == 1) {
                target_vis_val = vis_blur1d[target_index];
              }
              blurred_vis_sum += weight * target_vis_val;
              sum_weight += weight;
            }
          }
      }
    }
}

RT_PROGRAM void occlusion_filter_first_pass() {
  float3 cur_vis = vis[launch_index];
  float wxf = computeWxf(slope[launch_index].y);
  float blurred_vis = cur_vis.x;

  float2 cur_slope = slope[launch_index];

  if (!use_filter_n[launch_index] || !use_filter_occ[launch_index]) {
    vis_blur1d[launch_index] = blurred_vis;
    return;
  }

  if (blur_occ) {

    float blurred_vis_sum = 0.0f;
    float sum_weight = 0.0f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];
    size_t2 buf_size = vis.size();

    for (int i = -pixel_radius.x; i < pixel_radius.x; i++) {
      occlusionFilter(blurred_vis_sum, sum_weight, cur_world_loc, cur_n, wxf,
        launch_index.x+i, launch_index.y, buf_size, 0);
    }

    if (sum_weight > 0.0001f)
      blurred_vis = blurred_vis_sum / sum_weight;
  }

  vis_blur1d[launch_index] = blurred_vis;
}

RT_PROGRAM void occlusion_filter_second_pass() {
  float3 cur_vis = vis[launch_index];
  float wxf = computeWxf(slope[launch_index].y);
  float blurred_vis = vis_blur1d[launch_index];

  if (blur_occ) {
    if (!use_filter_occ[launch_index] || !use_filter_n[launch_index]) {
      vis[launch_index].x = blurred_vis;
      return;
    }

    float blurred_vis_sum = 0.0f;
    float sum_weight = 0.0f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];
    size_t2 buf_size = vis.size();

    for (int j = -pixel_radius.y; j < pixel_radius.y; j++) {
      occlusionFilter(blurred_vis_sum, sum_weight, cur_world_loc, cur_n, wxf,
        launch_index.x, launch_index.y+j, buf_size, 1);
    }

    if (sum_weight > 0.00001f)
      blurred_vis = blurred_vis_sum / sum_weight;
    else 
      blurred_vis = cur_vis.x;
  }

  vis[launch_index].x = blurred_vis;
}

__device__ __inline__ float2 s1s2FilterMaxMin(float2& cur_slope, bool& use_filt, int obj_id,
  unsigned int i, unsigned int j, const optix::size_t2& buf_size, unsigned int pass) {
    float2 output_slope = cur_slope;
    uint use_filter = 0;
    if (i > 0 && i < buf_size.x && j > 0 && j <buf_size.y) {
      uint2 target_index = make_uint2(i,j);
      float2 target_slope;
      int target_id = obj_id_b[target_index];
      if (target_id != obj_id)
        return output_slope;
      if (pass == 0) {
        target_slope = slope[target_index];
        use_filter = use_filter_occ[target_index];
      }
      else {
        target_slope = slope_filter1d[target_index];
        use_filter = use_filter_occ_filter1d[target_index];
      }
      if (use_filter) {
        output_slope.x = max(cur_slope.x, target_slope.x);
        output_slope.y = min(cur_slope.y, target_slope.y);
      }
    }
    use_filt |= use_filter;
    return output_slope;
}

#define S1S2_RADIUS 10

RT_PROGRAM void s1s2_filter_first_pass() {
  float2 cur_slope = slope[launch_index];
  size_t2 buf_size = slope.size();
  bool use_filter = use_filter_occ[launch_index];
  int obj_id = obj_id_b[launch_index];
  for (int i = -S1S2_RADIUS; i < S1S2_RADIUS; i++) {
    cur_slope = s1s2FilterMaxMin(cur_slope, use_filter, obj_id, launch_index.x + i, launch_index.y, buf_size, 0);
  }
  use_filter_occ_filter1d[launch_index] |= use_filter;
  slope_filter1d[launch_index] = cur_slope;
  return;
}

RT_PROGRAM void s1s2_filter_second_pass() {

  float2 cur_slope = slope_filter1d[launch_index];
  size_t2 buf_size = slope.size();
  bool use_filter = use_filter_occ_filter1d[launch_index];
  int obj_id = obj_id_b[launch_index];
  for (int i = -S1S2_RADIUS; i < S1S2_RADIUS; i++) {
    cur_slope = s1s2FilterMaxMin(cur_slope, use_filter, obj_id, launch_index.x, launch_index.y + i, buf_size, 1);
  }
  if (!use_filter_occ[launch_index]) {
    use_filter_occ[launch_index] |= use_filter;
    slope[launch_index] = cur_slope;
  }
  return;
}

//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.brdf = bg_color;
  prd_radiance.hit_shadow = false;
}

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()

{
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.hit = true;

  prd_shadow.distance_min = min(prd_shadow.distance_min, t_hit);
  prd_shadow.distance_max = max(prd_shadow.distance_max, t_hit);

  rtIgnoreIntersection();
}


//
// Phong surface shading with shadows
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(int,      obj_id, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

//asdf
rtBuffer<uint2, 2> shadow_rng_seeds;

RT_PROGRAM void closest_hit_radiance3()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_radiance.t_hit = t_hit;
  prd_radiance.world_loc = hit_point;
  prd_radiance.hit = true;
  prd_radiance.n = ffnormal;
  prd_radiance.obj_id = obj_id;

  uint2 seed = shadow_rng_seeds[launch_index];


  //Assume 1 light for now
  BasicLight light = lights[0];
  //float3 lc = light.color;
  float3 colorAvg = make_float3(0,0,0);

  //phong values
  float3 to_light = light.pos - hit_point;
  float dist_to_light = sqrt(to_light.x*to_light.x + to_light.y*to_light.y + to_light.z*to_light.z);
  prd_radiance.dist_to_light = dist_to_light;
  if(prd_radiance.first_pass) {
    float3 L = normalize(to_light);
    float nDl = max(dot( ffnormal, L ),0.0f);
    float3 H = normalize(L - ray.direction);
    float nDh = max(dot( ffnormal, H ),0.0f);
    //temporary - white light
    float3 Lc = make_float3(1,1,1);
    color += Kd * nDl * Lc;// * strength;
    if (nDh > 0)
      color += Ks * pow(nDh, phong_exp);
    prd_radiance.brdf = color;
  }

  /*
  //Stratify x
  for(int i=0; i<prd_radiance.sqrt_num_samples; ++i) {
    seed.x = rot_seed(seed.x, i);

    //Stratify y
    for(int j=0; j<prd_radiance.sqrt_num_samples; ++j) {
      seed.y = rot_seed(seed.y, j);

      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sample.x = (sample.x+((float)i))/prd_radiance.sqrt_num_samples;
      sample.y = (sample.y+((float)j))/prd_radiance.sqrt_num_samples;

      float3 target = (sample.x * lx + sample.y * ly) + lo;

      float strength = exp( -0.5 * ((light.pos.x - target.x) * (light.pos.x - target.x) \
        + (light.pos.y - target.y) * (light.pos.y - target.y) \
        + (light.pos.z - target.z) * (light.pos.z - target.z)) \
        / ( light_sigma * light_sigma));

      float3 sampleDir = normalize(target - hit_point);

      if(dot(ffnormal, sampleDir) > 0.0f) {
        prd_radiance.use_filter_n = true;
        prd_radiance.vis_weight_tot += strength;

        // SHADOW
        //cast ray and check for shadow
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(strength);
        shadow_prd.distance_max = 0;
        shadow_prd.distance_min = dist_to_light;
        shadow_prd.hit = false;
        optix::Ray shadow_ray ( hit_point, sampleDir, shadow_ray_type, 0.001);//scene_epsilon );
        rtTrace(top_shadower, shadow_ray, shadow_prd);

        if(shadow_prd.hit) {
          prd_radiance.hit_shadow = true;
          float d2min = dist_to_light - shadow_prd.distance_max;
          float d2max = dist_to_light - shadow_prd.distance_min;
          if (shadow_prd.distance_max < 0.000000001)
            d2min = dist_to_light;
          float s1 = dist_to_light/d2min - 1.0;
          float s2 = dist_to_light/d2max - 1.0;

          prd_radiance.s1 = max(prd_radiance.s1, s1);
          prd_radiance.s2 = min(prd_radiance.s2, s2);
        } else {
          prd_radiance.unavg_vis += strength;
        }
      }

    }
  }

  shadow_rng_seeds[launch_index] = seed;
  */

}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
