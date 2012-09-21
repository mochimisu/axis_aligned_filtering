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
rtDeclareVariable(PerRayData_indirect,   prd_indirect,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, indirect_ray_type , , );
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
//wxf= 1/b
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

//new gi stuff
rtBuffer<float3, 2>               direct_illum;
rtBuffer<float3, 2>               indirect_illum;
rtBuffer<float, 2>                z_perp_min;
rtBuffer<float3, 2>               indirect_illum_blur1d;
rtBuffer<float3, 2>               indirect_illum_filt;
rtBuffer<float4, 2>               indirect_illum_accum;

rtBuffer<float3, 2>               brdf;
//divided occlusion, undivided occlusion, wxf, num_samples
rtBuffer<float3, 2>               vis;
rtBuffer<float, 2>                vis_blur1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                spp;
rtBuffer<float, 2>                spp_cur;

rtBuffer<BasicLight>        lights;

rtDeclareVariable(uint,           frame, , );
rtDeclareVariable(uint,           filter_indirect, , );
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

RT_PROGRAM void pinhole_camera_initial_sample() {
  // Find direction to shoot ray in
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_radiance prd;

  if (frame == 0)
    indirect_illum_accum[launch_index] = make_float4(0);

  // Initialize the stuff we use in later passes
  brdf[launch_index] = make_float3(0,0,0);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);
  prd.sqrt_num_samples = normal_rpp;
  prd.hit = false;
  prd.zpmin = 100000000;

  rtTrace(top_object, ray, prd);

  z_perp_min[launch_index] = prd.zpmin;

  world_loc[launch_index] = prd.world_loc;
  direct_illum[launch_index] = prd.direct;
  indirect_illum[launch_index] = prd.indirect;
  n[launch_index] = normalize(prd.n);
}



RT_PROGRAM void display_camera() {

  if (brdf[launch_index].x < -1.0f) {
    output_buffer[launch_index] = make_color( bg_color );
    return;
  }

  output_buffer[launch_index] = make_color( direct_illum[launch_index]+indirect_illum[launch_index]);
  if (filter_indirect)
    output_buffer[launch_index] = make_color( direct_illum[launch_index]+indirect_illum_filt[launch_index]);

  if (view_mode) {
    if (view_mode == 1)
      output_buffer[launch_index] = make_color( direct_illum[launch_index] );
    if (view_mode == 2) 
    {
      if (filter_indirect)
        output_buffer[launch_index] = make_color( indirect_illum_filt[launch_index]);
      else
        output_buffer[launch_index] = make_color( indirect_illum[launch_index]);
    }
    if (view_mode == 3)
      output_buffer[launch_index] = make_color( heatMap(z_perp_min[launch_index]/200. ));
  }
}

__device__ __inline__ void indirectFilter( 
    float3& blurred_indirect_sum,
    float& sum_weight,
    const float3& cur_world_loc,
    float3 cur_n,
    float cur_zpmin,
    int i,
    int j,
    const size_t2& buf_size,
    unsigned int pass)
{
  const float dist_scale_threshold = 10.0f;
  const float dist_threshold = 100.0f;
  const float angle_threshold = 20.0f * M_PI/180.0f;

  if (i > 0 && i < buf_size.x && j > 0 && j < buf_size.y) {
    uint2 target_index = make_uint2(i,j);
    float3 target_indirect = indirect_illum[target_index];
    if (pass == 1)
      target_indirect = indirect_illum_blur1d[target_index];
    float target_zpmin = z_perp_min[target_index];
    float3 target_n = n[target_index];
    //bool use_filt = use_filt_indirect[target_index];
    bool use_filt = true;

    if (use_filt 
        && acos(dot(target_n, cur_n)) < angle_threshold
        && (1/target_zpmin - 1/cur_zpmin) < dist_scale_threshold
        )
    {
      float3 target_loc = world_loc[target_index];
      float3 diff = cur_world_loc - target_loc;
      float euclidean_distsq = dot(diff,diff);

      if (euclidean_distsq < (dist_threshold*dist_threshold))
      {
        float weight = gaussFilter(euclidean_distsq, 1/cur_zpmin);

        blurred_indirect_sum += weight * target_indirect;
        sum_weight += weight;
      }
    }
    
  }
}

RT_PROGRAM void indirect_filter_first_pass()
{
  float3 cur_indirect = indirect_illum[launch_index];
  float cur_zpmin = z_perp_min[launch_index];
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect = cur_indirect;

  float3 blurred_indirect_sum = make_float3(0.);
  float sum_weight = 0.f;

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  for (int i = -pixel_radius.x; i < pixel_radius.x; i++) 
  {
    indirectFilter(blurred_indirect_sum, sum_weight,
        cur_world_loc, cur_n, cur_zpmin, launch_index.x+i, launch_index.y,
        buf_size, 0);
  }

  if (sum_weight > 0.0001f)
    blurred_indirect = blurred_indirect_sum / sum_weight;
  indirect_illum_blur1d[launch_index] = blurred_indirect;
}

RT_PROGRAM void indirect_filter_second_pass()
{
  float3 cur_indirect = indirect_illum_blur1d[launch_index];
  float cur_zpmin = z_perp_min[launch_index];
  size_t2 buf_size = indirect_illum_blur1d.size();
  float3 blurred_indirect = cur_indirect;

  float3 blurred_indirect_sum = make_float3(0.);
  float sum_weight = 0.f;

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  for (int j = -pixel_radius.y; j < pixel_radius.y; j++) 
  {
    indirectFilter(blurred_indirect_sum, sum_weight,
        cur_world_loc, cur_n, cur_zpmin, launch_index.x, launch_index.y+j,
        buf_size, 1);
  }

  if (sum_weight > 0.0001f)
    blurred_indirect = blurred_indirect_sum / sum_weight;
  indirect_illum_filt[launch_index] = blurred_indirect;
}

//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_radiance.direct = bg_color;
  prd_radiance.indirect = make_float3(0);
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


//
// Calculates indirect color
//
RT_PROGRAM void any_hit_indirect()
{
  prd_indirect.color = make_float3(1);
  prd_indirect.hit = true;

  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;
  float3 hit_point = ray.origin + t_hit * ray.direction;

  prd_indirect.distance = t_hit;

  //Assume 1 light for now
  BasicLight light = lights[0];

  //phong values
  float3 to_light = light.pos - hit_point;
  float dist_to_light = sqrt(to_light.x*to_light.x + to_light.y*to_light.y + to_light.z*to_light.z);
  float3 L = normalize(to_light);
  float nDl = max(dot( ffnormal, L ),0.0f);
  float3 H = normalize(L - ray.direction);
  float nDh = max(dot( ffnormal, H ),0.0f);
  //temporary - white light
  float3 Lc = make_float3(1,1,1);
  color += Kd * nDl * Lc;// * strength;
  if (nDh > 0)
    color += Ks * pow(nDh, phong_exp);
  prd_indirect.color = color;

}

//Random direction buffer
rtBuffer<uint2, 2> indirect_rng_seeds;

//Rays from camera
RT_PROGRAM void closest_hit_radiance()
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

  uint2 seed = indirect_rng_seeds[launch_index];


  //Assume 1 light for now
  BasicLight light = lights[0];
  //float3 lc = light.color;
  float3 colorAvg = make_float3(0,0,0);

  //phong values
  float3 to_light = light.pos - hit_point;
  float3 L = normalize(to_light);
  float nDl = max(dot( ffnormal, L ),0.0f);
  float3 H = normalize(L - ray.direction);
  float nDh = max(dot( ffnormal, H ),0.0f);
  //temporary - white light
  float3 Lc = make_float3(1,1,1);
  color += Kd * nDl * Lc;// * strength;
  if (nDh > 0)
    color += Ks * pow(nDh, phong_exp);
  prd_radiance.direct = color;

  //indirect values
  int sample_sqrt = 4;
  float3 u,v,w;
  float3 sampleDir;
  createONB(ffnormal, u,v,w);
  float4 curIndAccum = indirect_illum_accum[launch_index];
  float3 indirectColor = make_float3(curIndAccum.x, curIndAccum.y, curIndAccum.z);
  float z_perp_min = 100000000;
  float2 sample = make_float2(0);
  //stratify
  for(int i=0; i<sample_sqrt; ++i) {
    seed.x = rot_seed(seed.x, i);
    sample.x = (sample.x+((float)i))/sample_sqrt;
    for(int j=0; j<sample_sqrt; ++j) {
      seed.y = rot_seed(seed.y, j);
      sample.y = (sample.y+((float)j))/sample_sqrt;
      float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sampleUnitHemisphere( sample, u,v,w, sampleDir);

      //construct indirect sample
      PerRayData_indirect indirect_prd;
      indirect_prd.hit = false;
      indirect_prd.color = make_float3(0,0,0);
      indirect_prd.distance = 100000000;


      optix::Ray indirect_ray ( hit_point, sampleDir, indirect_ray_type, 0.001);//scene_epsilon );

      rtTrace(top_shadower, indirect_ray, indirect_prd);

      if(indirect_prd.hit) {
        float3 zvec = indirect_prd.distance*sampleDir;
        float cur_zpmin = sqrt(dot(zvec,zvec) - dot(ffnormal,zvec));
        z_perp_min = min(z_perp_min, cur_zpmin);
      }

      //nDl term needed if sampling by cosine density?
      //float nDl = max(dot( ffnormal, normalize(sampleDir)), 0.f);

      indirectColor += Kd * indirect_prd.color;
      
    }
  }

  float num = indirect_illum_accum[launch_index].w \
              + (sample_sqrt*sample_sqrt);


  indirect_illum_accum[launch_index].w = num;
  indirect_illum_accum[launch_index].x = indirectColor.x;
  indirect_illum_accum[launch_index].y = indirectColor.y;
  indirect_illum_accum[launch_index].z = indirectColor.z;

  prd_radiance.indirect = indirectColor/num;
  prd_radiance.zpmin = z_perp_min;

  indirect_rng_seeds[launch_index] = seed;

}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
