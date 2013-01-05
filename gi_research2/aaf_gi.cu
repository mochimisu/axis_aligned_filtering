#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "aaf_gi.h"
#include "random.h"

using namespace optix;

struct PerRayData_pathtrace_shadow
{
  bool inShadow;
};

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );

// For camera
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtBuffer<float4, 2>              output_buffer;
rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

__device__ inline float3 powf(float3 a, float exp)
{
  return make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
}

// For miss program
rtDeclareVariable(float3,       bg_color, , );

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_float4(bad_color, 0.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void miss()
{
}


rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload,);

RT_PROGRAM void shadow()
{
  current_prd_shadow.inShadow = true;
  rtTerminateRay();
}


// AAF GI
//=========
struct PerRayData_direct
{
  float3 world_loc;
  float3 incoming_diffuse_light;
  float3 incoming_specular_light;
  float3 norm;
  float3 Kd;
  float3 Ks;
  float phong_exp;
  float z_dist;
  bool hit;
};
rtBuffer<float3, 2>               direct_illum;
rtBuffer<float3, 2>               indirect_illum;
rtBuffer<float3, 2>               indirect_illum_filter1d;
//specular buffers
rtBuffer<float3, 2>               indirect_illum_spec;
rtBuffer<float3, 2>               indirect_illum_spec_filter1d;

rtBuffer<float3, 2>               Kd_image;
rtBuffer<float3, 2>               Ks_image;
rtBuffer<float, 2>                phong_exp_image;
rtBuffer<float2, 2>               z_dist;
rtBuffer<float2, 2>               z_dist_filter1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                depth;
rtBuffer<float, 2>                spec_wvmax;
rtBuffer<char, 2>                 visible;
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,   phong_exp, , );
rtDeclareVariable(uint,  direct_ray_type, , );
rtDeclareVariable(uint,  num_buckets, , );
rtDeclareVariable(int,  z_filter_radius, , );
rtDeclareVariable(float,  vfov, , );
rtDeclareVariable(uint, max_spb_pass, , );
rtDeclareVariable(uint, indirect_ray_depth, , );
rtDeclareVariable(int, pixel_radius, , );
rtDeclareVariable(uint, use_textures, , );
rtDeclareVariable(float, spp_mu, , );
rtDeclareVariable(float, imp_samp_scale_diffuse, ,);
rtDeclareVariable(float, imp_samp_scale_specular, ,);

rtDeclareVariable(uint, view_mode, , );
rtDeclareVariable(uint, view_bucket, , );
rtDeclareVariable(float, max_heatmap, , );

rtBuffer<float, 2>                 target_spb_theoretical;
rtBuffer<float, 2>                 target_spb;

rtBuffer<float, 2>                 target_spb_spec_theoretical;
rtBuffer<float, 2>                 target_spb_spec;

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2>   diffuse_map;  
rtTextureSampler<float4, 2>   specular_map;  

//prefilter pass
//x: number of rejected pixels
//y: number of possible filtered pixels
rtBuffer<uint2, 2>                prefilter_rejected_filter1d;
rtBuffer<uint2, 2>                prefilter_rejected;


//omegaxf and sampling functions
__device__ __inline__ float glossy_blim(float3& r, float3& c, float m){
float vc = tan(acos(dot(r,c))); //r, c must be unit vectors
float        p00 =       2.017;
float        p10 =      0.2575;
float        p01 =       0.537;
float        p20 =    -0.05937;
float        p11 =     -0.1894;
float        p02 =    -0.03605;
float        p30 =    0.005778;
float        p21 =     0.03843;
float        p12 =    0.001897;
float        p03 =    0.001631;
float        p40 =  -0.0002033;
float        p31 =   -0.003547;
float        p22 =  -0.0002067;
float        p13 = -1.068e-005;
float        p04 = -3.413e-005;
float        p41 =   0.0001219;
float        p32 =  7.471e-006;
float        p23 =  6.913e-007;
float        p14 = -7.882e-009;
float        p05 =  2.601e-007;

float vc2 = vc*vc;
float vc3 = vc2*vc;
float vc4 = vc2*vc2;
float m2 = m*m;
float m3 = m2*m;
float m4 = m2*m2;
float m5 = m4*m;

return (p00 + p10*vc + p01*m + p20*vc2 + p11*vc*m + p02*m2 + p30*vc3 
    + p21*vc2*m + p12*vc*m2 + p03*m3 + p40*vc4 + p31*vc3*m + p22*vc2*m2  
    + p13*vc*m3 + p04*m4 + p41*vc4*m + p32*vc3*m2 + p23*vc2*m3
    + p14*vc*m4 + p05*m5);
}

// sample hemisphere with cosine^n density
__device__ __inline__ void sampleUnitHemispherePower( const optix::float2& sample,
  const optix::float3& U,
  const optix::float3& V,
  const optix::float3& W,
  const float power,
  optix::float3& point)
{
  using namespace optix;

  float phi = 2.f * M_PIf*sample.x;
  float r = sqrt(1 - pow(sample.y, 2.f/(power+1)));
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = pow(sample.y, 1.f/(power+1));

  point = x*U + y*V + z*W;
}



//Filter functions
// Our Gaussian Filter, based on w_xf
//wxf= 1/b
__device__ __inline__ float gaussFilter(float distsq, float wxf)
{
  float sample = distsq*wxf*wxf*spp_mu*spp_mu;
  if (sample > 0.9999) {
    return 0.0;
  }

  return exp(-sample);
}


__device__ __inline__ bool indirectFilterThresholds(
    const float3& cur_world_loc,
    const float3& cur_n,
    const float cur_zpmin,
    const uint2& target_index,
    const size_t2& buf_size,
    unsigned int bucket)
{
  const float z_threshold = .7f;
  //const float dist_threshold = 100.f;
  const float angle_threshold = 5.f * M_PI/180.f;
  //const float dist_threshold_sq = dist_threshold * dist_threshold;
  const float dist_threshold_sq = cur_zpmin*cur_zpmin;



  uint2 target_bucket_index = make_uint2(target_index.x,
      target_index.y*num_buckets+bucket);
  float target_zpmin = z_dist[target_bucket_index].x;
  float3 target_n = n[target_index];
  bool use_filt = visible[target_index];

  if (use_filt
      && abs(acos(dot(target_n, cur_n))) < angle_threshold
      //&& abs((1./target_zpmin - 1./cur_zpmin)/(1./target_zpmin 
      //   + 1./cur_zpmin)) < z_thres
      //&& abs((target_zpmin - cur_zpmin)/(target_zpmin + cur_zpmin)) < z_thres
      //&& (1/target_zpmin - 1/cur_zpmin) < dist_scale_threshold
     )
  {
    float3 target_loc = world_loc[target_index];
    float3 diff = cur_world_loc - target_loc;
    float euclidean_distsq = dot(diff,diff);

    if (euclidean_distsq < dist_threshold_sq)
    {
      return true;
    }
  }
  return false;
}

__device__ __inline__ float filterWeight(
    float proj_distsq, const float3& target_n, const float3& cur_n,
    float cur_zpmin, float wvmax)
{
  return gaussFilter(proj_distsq,
      wvmax*(1.f+50.f*acos(dot(target_n,cur_n)))/cur_zpmin);
}


__device__ __inline__ void indirectFilter( 
    float3& blurred_indirect_sum,
    float& sum_weight,
    float3& blurred_indirect_spec_sum,
    float& sum_weight_spec,
    const float cur_spec_wvmax,
    const float3& cur_world_loc,
    float3 cur_n,
    float cur_zpmin,
    const uint2& target_index,
    const size_t2& buf_size,
    unsigned int pass,
    unsigned int bucket)
{

  bool can_filter = false;
  if (target_index.x > 0 && target_index.x < buf_size.x 
      && target_index.y > 0 && target_index.y < buf_size.y)
  {
    can_filter = indirectFilterThresholds(
      cur_world_loc, cur_n, cur_zpmin, target_index, buf_size, bucket);
  }
  if (can_filter)
  {
    //TODO: cleanup
    uint2 target_bucket_index = make_uint2(target_index.x,
        target_index.y*num_buckets+bucket);

    float3 target_loc = world_loc[target_index];
    float3 diff = cur_world_loc - target_loc;
    float euclidean_distsq = dot(diff,diff);
    float rDn = dot(diff, cur_n);
    float proj_distsq = euclidean_distsq - rDn*rDn;
    float3 target_n = n[target_index];

    float diff_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
        2.);
    float spec_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
        cur_spec_wvmax);

    float3 target_indirect = indirect_illum[target_bucket_index];
    float3 target_indirect_spec = indirect_illum_spec[target_bucket_index];
    if (pass == 1)
    {
      target_indirect = indirect_illum_filter1d[target_bucket_index];
      target_indirect_spec = indirect_illum_spec_filter1d[target_bucket_index];
    }

    blurred_indirect_sum += diff_weight * target_indirect;
    sum_weight += diff_weight;

    blurred_indirect_spec_sum += spec_weight * target_indirect_spec;
    sum_weight_spec += spec_weight;
  }
}


//Ray Hit Programs
rtDeclareVariable(PerRayData_direct, prd_direct, rtPayload, );
RT_PROGRAM void closest_hit_direct()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD,
        geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD,
        shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction,
      world_geo_normal );

  prd_direct.hit = true;
  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_direct.z_dist = t_hit;
  prd_direct.world_loc = hit_point;
  prd_direct.norm = ffnormal;
  float3 cur_Kd = Kd;
  float3 cur_Ks = Ks;
  if (use_textures)
  {
    float2 uv = make_float2(texcoord);
    cur_Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
    cur_Ks = make_float3(tex2D(specular_map, uv.x, uv.y));
  }

  prd_direct.Kd = cur_Kd;
  prd_direct.Ks = cur_Ks;
  prd_direct.phong_exp = phong_exp;

  //lights
  unsigned int num_lights = lights.size();
  float3 diffuse = make_float3(0.0f);
  float3 specular = make_float3(0.0f);
  for(int i = 0; i < num_lights; ++i)
  {
    ParallelogramLight light = lights[i];
    float3 light_pos = light.corner;
    float Ldist = length(light_pos - hit_point);
    float3 L = normalize(light_pos - hit_point);
    float nDl = max(dot(ffnormal, L),0.f);
    float A = length(cross(light.v1, light.v2));
    float LnDl = dot( light.normal, L );


    float3 R = normalize(2*ffnormal*dot(ffnormal,L)-L);
    float nDr = max(dot(-ray.direction, R), 0.f);

    // cast shadow ray
    if ( nDl > 0.0f )
    {
      PerRayData_pathtrace_shadow shadow_prd;
      shadow_prd.inShadow = false;
      Ray shadow_ray = make_Ray( hit_point, L, pathtrace_shadow_ray_type, 
          scene_epsilon, Ldist );
      rtTrace(top_object, shadow_ray, shadow_prd);

      if(!shadow_prd.inShadow)
      {
        diffuse += make_float3(nDl);
        specular += make_float3(pow(nDr,phong_exp));
      }
    }

  }

  prd_direct.incoming_diffuse_light = diffuse;
  prd_direct.incoming_specular_light = specular;
}

RT_PROGRAM void closest_hit_indirect()
{
}


// First initial sample of the scene, record direct illum and such
RT_PROGRAM void sample_direct_z()
{
  size_t2 screen = direct_illum.size();

  float3 ray_origin = eye;
  float2 d = make_float2(launch_index)/make_float2(screen) * 2.f - 1.f;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  uint2 base_bucket_index = make_uint2(launch_index.x,
      launch_index.y*num_buckets);

  //initialize some buffers
  prefilter_rejected[launch_index] = make_uint2(0,1);
  for(int bucket = 0; bucket < num_buckets; ++bucket)
  {
    uint2 bucket_index = make_uint2(base_bucket_index.x,
        base_bucket_index.y+bucket);
    indirect_illum[bucket_index] = make_float3(0);
    indirect_illum_spec[bucket_index] = make_float3(0);
  }

  PerRayData_direct dir_samp;
  dir_samp.hit = false;
  Ray ray = make_Ray(ray_origin, ray_direction, direct_ray_type, 
      scene_epsilon, RT_DEFAULT_MAX);
  rtTrace(top_object, ray, dir_samp);
  if (!dir_samp.hit) {
    direct_illum[launch_index] = make_float3(0.34f,0.55f,0.85f);
    visible[launch_index] = false;
    for(int bucket = 0; bucket < num_buckets; ++bucket)
    {
      uint2 bucket_index = make_uint2(base_bucket_index.x,
          base_bucket_index.y+bucket);
      z_dist[bucket_index] = make_float2(1000000000000.f,0.f);
    }
    return;
  }
  visible[launch_index] = true;
  
  world_loc[launch_index] = dir_samp.world_loc;
  direct_illum[launch_index] = dir_samp.incoming_diffuse_light * dir_samp.Kd
    + dir_samp.incoming_specular_light * dir_samp.Ks;
  n[launch_index] = dir_samp.norm;
  Kd_image[launch_index] = dir_samp.Kd;
  Ks_image[launch_index] = dir_samp.Ks;
  depth[launch_index] = dir_samp.z_dist;


  //assuming 1 light for specular omegavmax calculation
  ParallelogramLight light = lights[0];
  float3 L = normalize(light.corner - world_loc[launch_index]);
  float3 cur_n = n[launch_index];
  float3 R = normalize(2*cur_n*dot(cur_n,L)-L);
  float3 to_camera = -ray_direction;

  //use R with light or R from camera?
  float3 perf_refl_r = ray_direction - 2*cur_n * dot(cur_n, ray_direction);
  float cur_spec_wvmax = glossy_blim(perf_refl_r, to_camera, 
      dir_samp.phong_exp);
  phong_exp_image[launch_index] = dir_samp.phong_exp;
  spec_wvmax[launch_index] = cur_spec_wvmax;


  int initial_bucket_samples_sqrt = 2;
  int initial_bucket_samples = initial_bucket_samples_sqrt
    * initial_bucket_samples_sqrt;


  float3 n_u, n_v, n_w;
  createONB(dir_samp.norm, n_u, n_v, n_w);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
      frame_number);

  for(int bucket = 0; bucket < num_buckets; ++bucket)
  {
    uint2 bucket_index = make_uint2(base_bucket_index.x,
        base_bucket_index.y+bucket);
    float bucket_zmin_dist = 1000000000000.f;
    float bucket_zmax_dist = scene_epsilon;
    for(int samp = 0; samp < initial_bucket_samples; ++samp)
    {
      float3 sample_dir;
      float2 rand_samp = make_float2(rnd(seed),rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%initial_bucket_samples_sqrt + rand_samp.x)
        /initial_bucket_samples_sqrt;
      rand_samp.y = (((int)samp/initial_bucket_samples_sqrt) + rand_samp.y)
        /initial_bucket_samples_sqrt;

      //vary theta component of sampling to sample split hemisphere
      rand_samp.x = (bucket + rand_samp.x)/num_buckets;
      //dont accept "grazing" angles
      rand_samp.y *= 0.95f;
      sampleUnitHemisphere(rand_samp, n_u, n_v, n_w, sample_dir);

      PerRayData_direct indir_samp;
      indir_samp.hit = false;
      Ray indir_ray = make_Ray(dir_samp.world_loc, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, indir_ray, indir_samp);
      if (indir_samp.hit)
      {
        bucket_zmin_dist = min(bucket_zmin_dist, indir_samp.z_dist);
        bucket_zmax_dist = max(bucket_zmax_dist, indir_samp.z_dist);
      }
    }
    z_dist[bucket_index] = make_float2(bucket_zmin_dist, bucket_zmax_dist);
  }
}
RT_PROGRAM void z_filter_first_pass()
{
    float2 cur_zd = z_dist[launch_index];
    float cur_zmin = cur_zd.x;
    float cur_zmax = cur_zd.y;
    for(int w=-z_filter_radius; w<=z_filter_radius; ++w)
    {
      float target_x = launch_index.x+w;
      float target_y = launch_index.y;
      if (target_x > 0 && target_x < z_dist.size().x)
      {
        uint2 target_index = make_uint2(target_x,target_y);
        float2 target_zd = z_dist[target_index];
        cur_zmin = min(cur_zmin, target_zd.x);
        cur_zmax = min(cur_zmax, target_zd.y);
      }
    }
    z_dist_filter1d[launch_index] = make_float2(cur_zmin, cur_zmax);
}
RT_PROGRAM void z_filter_second_pass()
{
    float2 cur_zd = z_dist_filter1d[launch_index];
    float cur_zmin = cur_zd.x;
    float cur_zmax = cur_zd.y;
    for(int h=-z_filter_radius; h<=z_filter_radius; ++h)
    {
      float target_x = launch_index.x;
      float target_y = launch_index.y+(num_buckets*h);
      if (target_y > 0 && target_y < z_dist.size().y)
      {
        uint2 target_index = make_uint2(target_x,target_y);
        float2 target_zd = z_dist_filter1d[target_index];
        cur_zmin = min(cur_zmin, target_zd.x);
        cur_zmax = min(cur_zmax, target_zd.y);
      }
    }
    z_dist[launch_index] = make_float2(cur_zmin, cur_zmax);
}

RT_PROGRAM void sample_indirect()
{
  size_t2 bucket = indirect_illum.size();
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;


  if(!visible[screen_index])
  {
    target_spb_theoretical[launch_index] = 0;
    target_spb[launch_index] = 0;
    target_spb_spec_theoretical[launch_index] = 0;
    target_spb_spec[launch_index] = 0;
    indirect_illum[launch_index] = make_float3(0);
    return;
  }

  float3 Kd = Kd_image[screen_index];
  float3 Ks = Ks_image[screen_index];
  float cur_phong_exp = phong_exp_image[screen_index];
  
  //calculate SPP
  float2 cur_zd = z_dist[launch_index]; //x: zmin, y:zmax
  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[screen_index] 
    * tan(vfov/2.*M_PI/180.);
  float diff_wvmax = 2; //Constant for now (diffuse)
  float alpha = 1;
  float spp_term1 = proj_dist * diff_wvmax/cur_zd.x + alpha;
  float spp_term2 = 1+cur_zd.y/cur_zd.x;

  float spp = imp_samp_scale_diffuse
	*spp_term1*spp_term1 * diff_wvmax*diff_wvmax 
    * spp_term2*spp_term2;


  float cur_spec_wvmax = spec_wvmax[screen_index];
  float spp_spec_term1 = proj_dist * cur_spec_wvmax/cur_zd.x + alpha;

  float spec_spp = imp_samp_scale_specular
	* spp_spec_term1 * spp_spec_term1 
    * cur_spec_wvmax*cur_spec_wvmax
    * spp_term2*spp_term2;

  
  //account for split buckets
  spp /= num_buckets;
  spec_spp /= num_buckets;

  target_spb_theoretical[launch_index] = spp;
  target_spb_spec_theoretical[launch_index] = spec_spp;

  uint2 pf_rej = prefilter_rejected[launch_index];
  
  float rej_scale = (1.f + (float)pf_rej.x/pf_rej.y);

  float Kd_mag = length(Kd);
  float Ks_mag = length(Ks);
  float Kd_Ks_ratio = Kd_mag/(Kd_mag+Ks_mag);

  spp = max(
      min(spp / (1.f-(float)pf_rej.x/pf_rej.y), 
        spp_mu*(float)max_spb_pass),
      1.f) * Kd_Ks_ratio;
  spec_spp = max( min(spp, spp_mu*(float)max_spb_pass), 1.f)
    *(1-Kd_Ks_ratio); 
  //TODO: distribute samples according to kd to ks ratio, account for prefilt

  float spp_sqrt = sqrt(spp);
  int spp_sqrt_int = (int) ceil(spp_sqrt);
  int spp_int = spp_sqrt_int * spp_sqrt_int;
  target_spb[launch_index] = spp_int;

  float spp_spec_sqrt = sqrt(spec_spp);
  int spp_spec_sqrt_int = (int) ceil(spp_spec_sqrt);
  int spp_spec_int = spp_spec_sqrt_int * spp_spec_sqrt_int;
  target_spb_spec[launch_index] = spp_spec_int;

  float3 first_hit = world_loc[screen_index];
  float3 normal = n[screen_index];

  //diffuse
  //sample this hemisphere with cosine (aligned to normal) weighting
  float3 incoming_diff_indirect = make_float3(0);
  float3 incoming_spec_indirect = make_float3(0);
  unsigned int seed = tea<16>(bucket.x*launch_index.y+launch_index.x,
      frame_number); //TODO :verify
  for (int samp = 0; samp < spp_int; ++samp)
  {
    PerRayData_direct prd;
    float3 ray_origin = first_hit;
    float3 ray_n = normal;
    float3 rn_u, rn_v, rn_w;
    float3 sample_dir;
    float3 sample_diffuse_color = make_float3(0);
    float3 sample_specular_color = make_float3(0);
    float3 prev_dir = normalize(first_hit-eye);
	float3 prev_Kd = make_float3(1.f);
	float3 prev_Ks = make_float3(1.f);
    float prev_phong_exp = cur_phong_exp;
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
      prd.hit = false;
      createONB(ray_n, rn_u, rn_v, rn_w);
      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%spp_sqrt_int + rand_samp.x)/spp_sqrt_int;
      rand_samp.y = (((int)samp/spp_sqrt_int) + rand_samp.y)/spp_sqrt_int;

      //sample inside appropriate bucket
      rand_samp.x = (cur_bucket + rand_samp.x)/num_buckets;
      sampleUnitHemisphere(rand_samp, rn_u, rn_v, rn_w, sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if (!prd.hit)
        break;
      float3 R = normalize(2*ray_n*dot(ray_n, sample_dir)-sample_dir);
      float nDr = max(dot(-prev_dir, R), 0.f);
      float3 incoming_light = prd.incoming_specular_light * prd.Ks 
        + prd.incoming_diffuse_light * prd.Kd;

      sample_diffuse_color += incoming_light * prev_Kd;
      sample_specular_color += incoming_light*pow(nDr, prev_phong_exp) 
		  * prev_Ks;
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
      prev_phong_exp = prd.phong_exp;
	  prev_Kd = prd.Kd;
	  prev_Ks = prd.Ks;
    }
    incoming_diff_indirect += sample_diffuse_color;
    incoming_spec_indirect += sample_specular_color;
  }

  //specular
  //sample this hemisphere with cos^n (aligned to reflected angle) weighting
  for (int samp = 0; samp < spp_spec_int; ++samp)
  {
    PerRayData_direct prd;
    float3 ray_origin = first_hit;
    float3 ray_n = normal;
    float3 rns_u, rns_v, rns_w;
    float3 prev_dir = normalize(first_hit-eye);

    float3 sample_dir;
    float3 sample_diffuse_color = make_float3(0);
    float3 sample_specular_color = make_float3(0);
	float3 prev_Kd = make_float3(1.f);
	float3 prev_Ks = make_float3(1.f);
    float prev_phong_exp = cur_phong_exp;
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
      prd.hit = false;

      prev_dir = normalize(prev_dir);
      float3 perf_refl = normalize(prev_dir - 2*ray_n*dot(ray_n, prev_dir));
      createONB(perf_refl, rns_u, rns_v, rns_w);

      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%spp_spec_sqrt_int + rand_samp.x)/spp_spec_sqrt_int;
      rand_samp.y = (((int)samp/spp_spec_sqrt_int) + rand_samp.y)
        /spp_spec_sqrt_int;

      //sample inside appropriate bucket
      rand_samp.x = (cur_bucket + rand_samp.x)/num_buckets;
      sampleUnitHemispherePower(rand_samp, rns_u, rns_v, rns_w, cur_phong_exp,
          sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if (!prd.hit)
        break;

      //move this somewhere else...
      float nDl = max(dot(ray_n,sample_dir),0.f);
      float3 incoming_light = prd.incoming_specular_light * prd.Ks 
        + prd.incoming_diffuse_light * prd.Kd;
      float3 R = normalize(ray_n*2*dot(ray_n, sample_dir)-sample_dir);
      float nDr = max(dot(-prev_dir,R),0.f);
      float nDrn = pow(nDr, prev_phong_exp);
/*
      if(nDrn > 0.01)
        sample_diffuse_color += incoming_light * nDl/nDrn
          * 2.f/(prev_phong_exp+1);
*/
      sample_specular_color += prev_Ks * incoming_light * nDl 
        * 2.f/(prev_phong_exp+1);

      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
      prev_phong_exp = prd.phong_exp;
	  prev_Ks = prd.Ks;
	  prev_Kd = prd.Kd;
    }
    incoming_diff_indirect += sample_diffuse_color;
    incoming_spec_indirect += sample_specular_color;
  }
  //incoming_diff_indirect /= (float)spp_int+spp_spec_int;
  incoming_spec_indirect /= (float)spp_int+spp_spec_int;
  incoming_diff_indirect /= (float)spp_int;
  //incoming_spec_indirect /= (float)spp_int;
  
  indirect_illum[launch_index] = incoming_diff_indirect;
  indirect_illum_spec[launch_index] = incoming_spec_indirect;

}
RT_PROGRAM void indirect_filter_first_pass()
{
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 blurred_indirect_spec_sum = make_float3(0.f);
  float sum_weight_spec = 0.f;

  float cur_spec_wvmax = spec_wvmax[screen_index];

  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];

  if (visible[screen_index])
    for (int i = -pixel_radius; i < pixel_radius; ++i)
    {
      uint2 target_index = make_uint2(screen_index.x+i, screen_index.y);
      indirectFilter(blurred_indirect_sum, sum_weight,
          blurred_indirect_spec_sum, sum_weight_spec, cur_spec_wvmax,
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 0, cur_bucket);
    }

  if (sum_weight > 0.0001f)
    indirect_illum_filter1d[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum_filter1d[launch_index] = indirect_illum[launch_index];
  if (sum_weight_spec > 0.0001f)
    indirect_illum_spec_filter1d[launch_index] = blurred_indirect_spec_sum
      /sum_weight_spec;
  else
    indirect_illum_spec_filter1d[launch_index] = 
      indirect_illum_spec[launch_index];
}
RT_PROGRAM void indirect_filter_second_pass()
{
  
  size_t2 screen_size = output_buffer.size();
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 blurred_indirect_spec_sum = make_float3(0.f);
  float sum_weight_spec = 0.f;

  float cur_spec_wvmax = spec_wvmax[screen_index];


  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];
  
  float proj_dist = 2./screen_size.y * depth[screen_index] 
    * tan(vfov/2.*M_PI/180.);
  int radius = min(10.f,max(1.f,cur_zmin/proj_dist));
  radius = 10;
  
  if (visible[screen_index])
    for (int i = -radius; i < radius; ++i)
    {
      uint2 target_index = make_uint2(screen_index.x, screen_index.y+i);
      indirectFilter(blurred_indirect_sum, sum_weight,
          blurred_indirect_spec_sum, sum_weight_spec, cur_spec_wvmax,
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 1, cur_bucket);
    }

  if (sum_weight > 0.0001f)
    indirect_illum[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum[launch_index] = indirect_illum_filter1d[launch_index];
  if (sum_weight_spec > 0.0001f)
    indirect_illum_spec[launch_index] = blurred_indirect_spec_sum
      /sum_weight_spec;
  else
    indirect_illum_spec[launch_index] = 
      indirect_illum_spec_filter1d[launch_index];

}
RT_PROGRAM void indirect_prefilter_first_pass()
{
  
  size_t2 screen_size = output_buffer.size();
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();

  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];

  uint2 cur_prefilter_rej = make_uint2(0,0);

  float proj_dist = 2./screen_size.y * depth[screen_index] 
    * tan(vfov/2.*M_PI/180.);
  int radius = min(10.f,max(1.f,cur_zmin/proj_dist));
  radius = 10;

  if (visible[screen_index])
    for (int i = -radius; i < radius; ++i)
    {
      //TODO: cleanup
      uint2 target_index = make_uint2(screen_index.x, screen_index.y+i);
      if (target_index.x > 0 && target_index.x < buf_size.x 
          && target_index.y > 0 && target_index.y < buf_size.y)
      {
        bool can_filter = indirectFilterThresholds(
            cur_world_loc, cur_n, cur_zmin, target_index, 
            buf_size, cur_bucket);
        float3 target_loc = world_loc[target_index];
        float3 diff = cur_world_loc - target_loc;
        float euclidean_distsq = dot(diff,diff);
        float rDn = dot(diff, cur_n);
        float proj_distsq = euclidean_distsq - rDn*rDn;
        float3 target_n = n[target_index];

        float weight = filterWeight(proj_distsq, target_n, cur_n, cur_zmin, 
            2.);
        if (weight > 0.01)
        {
          cur_prefilter_rej.x += (!can_filter);
          cur_prefilter_rej.y += 1;
        }
      }
    }
  prefilter_rejected_filter1d[launch_index] = cur_prefilter_rej;
}

RT_PROGRAM void indirect_prefilter_second_pass()
{
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();

  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];

  uint2 cur_prefilter_rej = make_uint2(0,0);

  if (visible[screen_index])
    for (int i = -pixel_radius; i < pixel_radius; ++i)
    {
      //TODO: cleanup
      uint2 target_index = make_uint2(screen_index.x+i, screen_index.y);
      if (target_index.x > 0 && target_index.x < buf_size.x 
          && target_index.y > 0 && target_index.y < buf_size.y)
      {
        bool can_filter = indirectFilterThresholds(
            cur_world_loc, cur_n, cur_zmin, target_index, 
            buf_size, cur_bucket);
        float3 target_loc = world_loc[target_index];
        float3 diff = cur_world_loc - target_loc;
        float euclidean_distsq = dot(diff,diff);
        float rDn = dot(diff, cur_n);
        float proj_distsq = euclidean_distsq - rDn*rDn;
        float3 target_n = n[target_index];

        float weight = filterWeight(proj_distsq, target_n, cur_n, cur_zmin, 
            2.);
        if (weight > 0.01)
        {
          uint2 target_bucket_index = make_uint2(launch_index.x+i,
              launch_index.y);
          uint2 firstpass_pf_rej = 
            prefilter_rejected_filter1d[target_bucket_index];
          if (!can_filter)
            cur_prefilter_rej.x += firstpass_pf_rej.x;
          cur_prefilter_rej.y += firstpass_pf_rej.y;
        }
      }
    }
  prefilter_rejected[launch_index] = cur_prefilter_rej;
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

RT_PROGRAM void display()
{
  float3 indirect_illum_combined = make_float3(0);
  float3 indirect_illum_spec_combined = make_float3(0);
  for (int i = 0; i < num_buckets; ++i)
  {
    indirect_illum_combined += indirect_illum[make_uint2(launch_index.x, 
        launch_index.y*num_buckets+i)];
    indirect_illum_spec_combined += indirect_illum_spec[
      make_uint2(launch_index.x, launch_index.y*num_buckets+i)];
  }

  indirect_illum_combined /= num_buckets;
  float3 indirect_illum_full = indirect_illum_combined * Kd_image[launch_index];
  indirect_illum_spec_combined /= num_buckets;
  float3 indirect_illum_spec_full = indirect_illum_spec_combined 
    * Ks_image[launch_index];
  output_buffer[launch_index] = make_float4(
      direct_illum[launch_index] 
      + indirect_illum_spec_full + indirect_illum_full,1);

  //other view modes
  if (view_mode)
  {
    bool view_separated_bucket = (view_bucket > 0)
      && (view_bucket <= num_buckets);
    uint2 target_bucket_index = make_uint2(launch_index.x, 
        launch_index.y*num_buckets+view_bucket-1);
    if (view_mode == 1)
      output_buffer[launch_index] = make_float4(direct_illum[launch_index],1);
    if (view_mode == 2)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum[target_bucket_index] * Kd_image[launch_index]
            + indirect_illum_spec[target_bucket_index] * Ks_image[launch_index]
            ,1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_full+indirect_illum_spec_full,1);
    }
    if (view_mode == 3)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum[target_bucket_index]
            +indirect_illum_spec[target_bucket_index],1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_combined+indirect_illum_spec_combined,1);
    }
    if (view_mode == 4)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum[target_bucket_index] * Kd_image[launch_index],1);
      else
        output_buffer[launch_index] = make_float4(indirect_illum_full,1);
    }
    if (view_mode == 5)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum[target_bucket_index],1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_combined,1);
    }

    if (view_mode == 6)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            Ks_image[launch_index]*indirect_illum_spec[target_bucket_index],1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_spec_full,1);
    }
    if (view_mode == 7)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum_spec[target_bucket_index],1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_spec_combined,1);
    }

  }

}

RT_PROGRAM void display_heatmaps()
{
  //other view modes
  if (view_mode)
  {
    bool view_separated_bucket = (view_bucket > 0)
      && (view_bucket <= num_buckets);
    uint2 target_bucket_index = make_uint2(launch_index.x, 
        launch_index.y*num_buckets+view_bucket-1);
    if (view_mode == 8)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
          z_dist[target_bucket_index].x/max_heatmap));
      else
      {
        float z_min_combined = z_dist[make_uint2(launch_index.x,
            launch_index.y*num_buckets)].x;
        for (int i = 0; i < num_buckets; ++i)
        {
          z_min_combined = min(z_min_combined, 
              z_dist[make_uint2(launch_index.x,
                launch_index.y*num_buckets+i)].x);
        }
        output_buffer[launch_index] = make_float4(heatMap(z_min_combined/
              max_heatmap));
      }
    }
    if (view_mode == 9)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
          z_dist[target_bucket_index].y/max_heatmap));
      else
      {
        float z_max_combined = z_dist[make_uint2(launch_index.x,
            launch_index.y*num_buckets)].x;
        for (int i = 0; i < num_buckets; ++i)
        {
          z_max_combined = max(z_max_combined, 
              z_dist[make_uint2(launch_index.x,
                launch_index.y*num_buckets+i)].y);
        }
        output_buffer[launch_index] = make_float4(heatMap(z_max_combined
              /max_heatmap));

      }
    }
    if (view_mode == 10)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              target_spb[target_bucket_index]/max_heatmap));
      else
      {
        float combined_spp;
        for (int i = 0; i < num_buckets; ++i)
        {
          combined_spp += target_spb[make_uint2(launch_index.x,
                launch_index.y*num_buckets+i)];
        }
        output_buffer[launch_index] = make_float4(heatMap(combined_spp
              /(max_heatmap*num_buckets)));
      }

    }
    if (view_mode == 11)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              target_spb_theoretical[target_bucket_index]/max_heatmap));
      else
      {
        float combined_spp;
        for (int i = 0; i < num_buckets; ++i)
        {
          combined_spp += target_spb_theoretical[make_uint2(launch_index.x,
                launch_index.y*num_buckets+i)];
        }
        output_buffer[launch_index] = make_float4(heatMap(combined_spp
              /(max_heatmap*num_buckets)));
      }
    }
    if (view_mode == 12)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              (float)
              prefilter_rejected[target_bucket_index].x/
              prefilter_rejected[target_bucket_index].y/max_heatmap));
      else
      {
        uint2 combined_rejected = make_uint2(0,0);
        for (int i = 0; i < num_buckets; ++i)
        {
          uint2 cur_bucket_index = make_uint2(launch_index.x,
              launch_index.y*num_buckets+i);
          combined_rejected.x += prefilter_rejected[cur_bucket_index].x;
          combined_rejected.y += prefilter_rejected[cur_bucket_index].y;
        }
        output_buffer[launch_index] = make_float4(heatMap(
              (float)
              combined_rejected.x/combined_rejected.y/max_heatmap));
      }
    }
    if (view_mode == 13)
    {
      output_buffer[launch_index] = make_float4(heatMap(
            spec_wvmax[launch_index]/max_heatmap));
    }
    if (view_mode == 14)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              target_spb_spec[target_bucket_index]/max_heatmap));
      else
      {
        float combined_spp;
        for (int i = 0; i < num_buckets; ++i)
        {
          combined_spp += target_spb_spec[make_uint2(launch_index.x,
              launch_index.y*num_buckets+i)];
        }
        output_buffer[launch_index] = make_float4(heatMap(combined_spp
              /(max_heatmap*num_buckets)));
      }

    }
    if (view_mode == 15)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              target_spb_spec_theoretical[target_bucket_index]/max_heatmap));
      else
      {
        float combined_spp;
        for (int i = 0; i < num_buckets; ++i)
        {
          combined_spp += target_spb_spec_theoretical[
            make_uint2(launch_index.x, launch_index.y*num_buckets+i)];
        }
        output_buffer[launch_index] = make_float4(heatMap(combined_spp
              /(max_heatmap*num_buckets)));
      }
    }
  }

}

// Ground truth accumulative indirect sampling
rtDeclareVariable(uint, total_gt_samples_sqrt, , );
rtDeclareVariable(uint, gt_samples_per_pass, , );
rtDeclareVariable(uint, gt_pass, , );
rtDeclareVariable(uint, gt_total_pass, , );
RT_PROGRAM void sample_indirect_gt()
{
  int total_gt_samples = total_gt_samples_sqrt*total_gt_samples_sqrt;
  int samples_sofar = (float)total_gt_samples*gt_pass/gt_total_pass;
  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*num_buckets);
  uint2 screen_index = launch_index;

  if(!visible[screen_index])
  {
    indirect_illum[bucket_index] = make_float3(0);
    return;
  }

  if (gt_pass == 0)
    for (int i = 0; i<num_buckets; ++i)
    {
      uint2 cur_bucket = make_uint2(bucket_index.x, bucket_index.y+i);
      indirect_illum[cur_bucket] = make_float3(0);
    }

  int samp_this_pass = gt_samples_per_pass;
  if (gt_pass == gt_total_pass - 1)
  {
    samp_this_pass = total_gt_samples-samples_sofar;
  }

  float3 first_hit = world_loc[screen_index];
  float3 normal = n[screen_index];
  float3 Kd = Kd_image[screen_index];

  size_t2 out_buf = output_buffer.size();

  //ignore buckets, and place all results in the first bucket multiplied
  //by number of buckets used in non-gt code so we can use the same code
  float3 incoming_indirect_diffuse = make_float3(0);
  float3 incoming_indirect_specular = make_float3(0);
  unsigned int seed = tea<16>(out_buf.x*launch_index.y+launch_index.x,
      frame_number); //TODO :verify
  for(int samp = 0; samp < samp_this_pass; ++samp)
  {
    int global_samp = samples_sofar + samp;

    PerRayData_direct prd;
    float3 ray_origin = first_hit;
    float3 ray_n = normal;
    float3 rn_u, rn_v, rn_w;
    float3 sample_dir;
    float3 incoming_diffuse = make_float3(0);
    float3 incoming_specular = make_float3(0);
    float3 prev_dir = normalize(first_hit-eye);
    float prev_phong_exp = phong_exp_image[screen_index];
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
      prd.hit = false;
      createONB(ray_n, rn_u, rn_v, rn_w);
      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify based on pass and total # of passes
      rand_samp.x = (global_samp%total_gt_samples_sqrt + rand_samp.x)
        /total_gt_samples_sqrt;
      rand_samp.y = (((int)global_samp/total_gt_samples_sqrt) + rand_samp.y)
        /total_gt_samples_sqrt;

      sampleUnitHemisphere(rand_samp, rn_u, rn_v, rn_w, sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if (!prd.hit)
        break;
      float3 R = normalize(ray_n*2*dot(ray_n, sample_dir)-sample_dir);
      float nDr = max(dot(-prev_dir,R),0.f);
      float3 incoming_light = prd.incoming_diffuse_light * prd.Kd
        + prd.incoming_specular_light * prd.Ks;
      incoming_diffuse += incoming_light;
      incoming_specular += incoming_light * pow(nDr, prev_phong_exp);
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
      prev_phong_exp = prd.phong_exp;
    }
    incoming_indirect_diffuse += incoming_diffuse;
    incoming_indirect_specular += incoming_specular;

  }

  indirect_illum[bucket_index] += incoming_indirect_diffuse
    /total_gt_samples*num_buckets;
  indirect_illum_spec[bucket_index] += incoming_indirect_specular
    /total_gt_samples*num_buckets;

}
