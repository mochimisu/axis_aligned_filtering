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
  float3 norm;
  float3 Kd;
  float z_dist;
  bool hit;
};
rtBuffer<float3, 2>               direct_illum;
rtBuffer<float3, 2>               indirect_illum;
rtBuffer<float3, 2>               indirect_illum_filter1d;

rtBuffer<float3, 2>               Kd_image;
rtBuffer<float2, 2>               z_dist;
rtBuffer<float2, 2>               z_dist_filter1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                depth;
rtBuffer<float, 2>                target_spb_theoretical;
rtBuffer<float, 2>                target_spb;
rtBuffer<char, 2>                 visible;
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(uint,  direct_ray_type, , );
rtDeclareVariable(int,  z_filter_radius, , );
rtDeclareVariable(float,  vfov, , );
rtDeclareVariable(uint, max_spb_pass, , );
rtDeclareVariable(uint, indirect_ray_depth, , );
rtDeclareVariable(int, pixel_radius, , );
rtDeclareVariable(uint, use_textures, , );
rtDeclareVariable(float, spp_mu, , );
rtDeclareVariable(float, imp_samp_scale_diffuse, ,);

rtDeclareVariable(uint, view_mode, , );
rtDeclareVariable(float, max_heatmap, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2>   diffuse_map;  

//prefilter pass
//x: number of rejected pixels
//y: number of possible filtered pixels
rtBuffer<uint2, 2>                prefilter_rejected_filter1d;
rtBuffer<uint2, 2>                prefilter_rejected;

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





__device__ __inline__ bool indirectFilterThresholds(
    const float3& cur_world_loc,
    const float3& cur_n,
    const float cur_zpmin,
    const uint2& target_index,
    const size_t2& buf_size)
{
  const float z_threshold = .7f;
  const float angle_threshold = 5.f * M_PI/180.f;
  const float dist_threshold_sq = cur_zpmin*cur_zpmin;

  float target_zpmin = z_dist[target_index].x;
  float3 target_n = n[target_index];
  bool use_filt = visible[target_index];

  if (use_filt
      && abs(acos(dot(target_n, cur_n))) < angle_threshold
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

//Filter functions
// Our Gaussian Filter, based on w_xf
//wxf= 1/b
__device__ __inline__ float filterWeight(
    float proj_distsq, const float3& target_n, const float3& cur_n,
    float cur_zpmin)
{
	float wxf = 2.f*(1.f+50.f*acos(dot(target_n,cur_n)))/cur_zpmin;
	float sample = proj_distsq*wxf*wxf;
	if (sample > 0.9999) {
		return 0.0;
	}

	return exp(-sample);

}


__device__ __inline__ void indirectFilter( 
    float3& blurred_indirect_sum,
    float& sum_weight,
    const float3& cur_world_loc,
    float3 cur_n,
    float cur_zpmin,
    const uint2& target_index,
    const size_t2& buf_size,
    unsigned int pass)
{

  bool can_filter = false;
  if (target_index.x > 0 && target_index.x < buf_size.x 
      && target_index.y > 0 && target_index.y < buf_size.y)
  {
    can_filter = indirectFilterThresholds(
      cur_world_loc, cur_n, cur_zpmin, target_index, buf_size);
  }
  if (can_filter)
  {
    //TODO: cleanup

    float3 target_loc = world_loc[target_index];
    float3 diff = cur_world_loc - target_loc;
    float euclidean_distsq = dot(diff,diff);
    float rDn = dot(diff, cur_n);
    float proj_distsq = euclidean_distsq - rDn*rDn;
    float3 target_n = n[target_index];

    float diff_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin);


    float3 target_indirect = indirect_illum[target_index];
    if (pass == 1)
    {
      target_indirect = indirect_illum_filter1d[target_index];
    }

    blurred_indirect_sum += diff_weight * target_indirect;
    sum_weight += diff_weight;
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
  if (use_textures)
  {
    float2 uv = make_float2(texcoord);
    cur_Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
  }

  prd_direct.Kd = cur_Kd;

  //lights
  unsigned int num_lights = lights.size();
  float3 diffuse = make_float3(0.0f);
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
      }
    }

  }

  prd_direct.incoming_diffuse_light = diffuse;
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

  //initialize some buffers
  prefilter_rejected[launch_index] = make_uint2(0,1);

  indirect_illum[launch_index] = make_float3(0);

  PerRayData_direct dir_samp;
  dir_samp.hit = false;
  Ray ray = make_Ray(ray_origin, ray_direction, direct_ray_type, 
      scene_epsilon, RT_DEFAULT_MAX);
  rtTrace(top_object, ray, dir_samp);
  if (!dir_samp.hit) {
    direct_illum[launch_index] = make_float3(0.34f,0.55f,0.85f);
    visible[launch_index] = false;
	z_dist[launch_index] = make_float2(1000000000000.f,0.f);
    return;
  }
  visible[launch_index] = true;
  
  world_loc[launch_index] = dir_samp.world_loc;
  direct_illum[launch_index] = dir_samp.incoming_diffuse_light * dir_samp.Kd;
  n[launch_index] = dir_samp.norm;
  Kd_image[launch_index] = dir_samp.Kd;
  depth[launch_index] = dir_samp.z_dist;




  int initial_bucket_samples_sqrt = 2;
  int initial_bucket_samples = initial_bucket_samples_sqrt
    * initial_bucket_samples_sqrt;


  float3 n_u, n_v, n_w;
  createONB(dir_samp.norm, n_u, n_v, n_w);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
	  frame_number);

  float bucket_zmin_dist = 1000000000000.f;
  float bucket_zmax_dist = scene_epsilon;
  PerRayData_direct indir_samp;
  float3 sample_dir;
  for(int samp = 0; samp < initial_bucket_samples; ++samp)
  {
	  float2 rand_samp = make_float2(rnd(seed),rnd(seed));
	  //stratify x,y
	  rand_samp.x = (samp%initial_bucket_samples_sqrt + rand_samp.x)
		  /initial_bucket_samples_sqrt;
	  rand_samp.y = (((int)samp/initial_bucket_samples_sqrt) + rand_samp.y)
		  /initial_bucket_samples_sqrt;

	  //dont accept "grazing" angles
	  rand_samp.y *= 0.95f;
	  sampleUnitHemisphere(rand_samp, n_u, n_v, n_w, sample_dir);

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
  z_dist[launch_index] = make_float2(bucket_zmin_dist, bucket_zmax_dist);
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
      float target_y = launch_index.y+h;
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

  if(!visible[launch_index])
  {
    indirect_illum[launch_index] = make_float3(0);
    return;
  }

  float3 Kd = Kd_image[launch_index];
  
  //calculate SPP
  float2 cur_zd = z_dist[launch_index]; //x: zmin, y:zmax
  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] 
    * tan(vfov/2.*M_PI/180.);
  float diff_wvmax = 2; //Constant for now (diffuse)
  float alpha = 1;
  float spp_term1 = proj_dist * diff_wvmax/cur_zd.x + alpha;
  float spp_term2 = 1+cur_zd.y/cur_zd.x;

  float spp = imp_samp_scale_diffuse
	*spp_term1*spp_term1 * diff_wvmax*diff_wvmax 
    * spp_term2*spp_term2;
  target_spb_theoretical[launch_index] = spp;


  uint2 pf_rej = prefilter_rejected[launch_index];
  
  float rej_scale = (1.f + (float)pf_rej.x/pf_rej.y);


  spp = max(
      min(spp / (1.f-(float)pf_rej.x/pf_rej.y), 
        spp_mu*(float)max_spb_pass),
      1.f);

  float spp_sqrt = sqrt(spp);
  int spp_sqrt_int = (int) ceil(spp_sqrt);
  int spp_int = spp_sqrt_int * spp_sqrt_int;
  target_spb[launch_index] = (float)spp_int;

  float3 first_hit = world_loc[launch_index];
  float3 normal = n[launch_index];

  //diffuse
  //sample this hemisphere with cosine (aligned to normal) weighting
  float3 incoming_diff_indirect = make_float3(0);
  size_t2 screen = direct_illum.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
      frame_number); //TODO :verify
  float3 rn_u, rn_v, rn_w;
  float3 sample_dir;
  float3 ray_origin;
  float3 ray_n;
  float3 sample_diffuse_color;
  float3 prev_dir;
  float3 prev_Kd;

  float2 rand_samp;
  float3 R;
  float nDr;
  float3 incoming_light;

  int depth;

  for (int samp = 0; samp < spp_int; ++samp)
  {
    PerRayData_direct prd;
    ray_origin = first_hit;
    ray_n = normal;
    sample_diffuse_color = make_float3(0);
    prev_dir = normalize(first_hit-eye);
	prev_Kd = make_float3(1.f);
    for (depth = 0; depth < indirect_ray_depth; ++depth)
    {
      prd.hit = false;
      createONB(ray_n, rn_u, rn_v, rn_w);
      rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%spp_sqrt_int + rand_samp.x)/spp_sqrt_int;
      rand_samp.y = (((int)samp/spp_sqrt_int) + rand_samp.y)/spp_sqrt_int;

      sampleUnitHemisphere(rand_samp, rn_u, rn_v, rn_w, sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if (!prd.hit)
        break;
      R = normalize(2*ray_n*dot(ray_n, sample_dir)-sample_dir);
      nDr = max(dot(-prev_dir, R), 0.f);
      incoming_light = prd.incoming_diffuse_light * prd.Kd;

      sample_diffuse_color += incoming_light * prev_Kd;
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
	  prev_Kd = prd.Kd;
    }
    incoming_diff_indirect += sample_diffuse_color;
  }

  incoming_diff_indirect /= (float)spp_int;
  
  indirect_illum[launch_index] = incoming_diff_indirect;

}
RT_PROGRAM void indirect_filter_first_pass()
{
  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  if (visible[launch_index])
    for (int i = -pixel_radius; i < pixel_radius; ++i)
    {
      uint2 target_index = make_uint2(launch_index.x+i, launch_index.y);
      indirectFilter(blurred_indirect_sum, sum_weight,
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 0);
    }

  if (sum_weight > 0.0001f)
    indirect_illum_filter1d[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum_filter1d[launch_index] = indirect_illum[launch_index];
}
RT_PROGRAM void indirect_filter_second_pass()
{
  
  size_t2 screen_size = output_buffer.size();

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];
  
  float proj_dist = 2./screen_size.y * depth[launch_index] 
    * tan(vfov/2.*M_PI/180.);
  int radius = min(10.f,max(1.f,cur_zmin/proj_dist));
  
  if (visible[launch_index])
    for (int i = -radius; i < radius; ++i)
    {
      uint2 target_index = make_uint2(launch_index.x, launch_index.y+i);
      indirectFilter(blurred_indirect_sum, sum_weight,
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 1);
    }

  if (sum_weight > 0.0001f)
    indirect_illum[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum[launch_index] = indirect_illum_filter1d[launch_index];
}
RT_PROGRAM void indirect_prefilter_first_pass()
{
  
  size_t2 screen_size = output_buffer.size();

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  uint2 cur_prefilter_rej = make_uint2(0,0);

  float proj_dist = 2./screen_size.y * depth[launch_index] 
    * tan(vfov/2.*M_PI/180.);
  int radius = min(10.f,max(1.f,cur_zmin/proj_dist));

  if (visible[launch_index])
    for (int i = -radius; i < radius; ++i)
    {
      //TODO: cleanup
      uint2 target_index = make_uint2(launch_index.x, launch_index.y+i);
      if (target_index.x > 0 && target_index.x < buf_size.x 
          && target_index.y > 0 && target_index.y < buf_size.y)
      {
        bool can_filter = indirectFilterThresholds(
            cur_world_loc, cur_n, cur_zmin, target_index, 
            buf_size);
        float3 target_loc = world_loc[target_index];
        float3 diff = cur_world_loc - target_loc;
        float euclidean_distsq = dot(diff,diff);
        float rDn = dot(diff, cur_n);
        float proj_distsq = euclidean_distsq - rDn*rDn;
        float3 target_n = n[target_index];

		cur_prefilter_rej.x += (!can_filter);
		cur_prefilter_rej.y += 1;
      }
    }
  prefilter_rejected_filter1d[launch_index] = cur_prefilter_rej;
}

RT_PROGRAM void indirect_prefilter_second_pass()
{

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  uint2 cur_prefilter_rej = make_uint2(0,0);

  if (visible[launch_index])
    for (int i = -pixel_radius; i < pixel_radius; ++i)
    {
      //TODO: cleanup
      uint2 target_index = make_uint2(launch_index.x+i, launch_index.y);
      if (target_index.x > 0 && target_index.x < buf_size.x 
          && target_index.y > 0 && target_index.y < buf_size.y)
      {
        bool can_filter = indirectFilterThresholds(
            cur_world_loc, cur_n, cur_zmin, target_index, 
            buf_size);
        float3 target_loc = world_loc[target_index];
        float3 diff = cur_world_loc - target_loc;
        float euclidean_distsq = dot(diff,diff);
        float rDn = dot(diff, cur_n);
        float proj_distsq = euclidean_distsq - rDn*rDn;
        float3 target_n = n[target_index];

		uint2 firstpass_pf_rej = 
			prefilter_rejected_filter1d[target_index];
		if (!can_filter)
			cur_prefilter_rej.x += firstpass_pf_rej.x;
		cur_prefilter_rej.y += firstpass_pf_rej.y;
      }
    }
  prefilter_rejected[launch_index] = cur_prefilter_rej;
}

RT_PROGRAM void display()
{
  float3 indirect_illum_combined = indirect_illum[make_uint2(launch_index.x, 
	  launch_index.y)];
  float3 indirect_illum_full = indirect_illum_combined * Kd_image[launch_index];
  output_buffer[launch_index] = make_float4(
      direct_illum[launch_index] 
      + indirect_illum_full,1);
	  /*
	  //other view modes
	  if (view_mode)
	  {
	  if (view_mode == 1)
	  output_buffer[launch_index] = make_float4(direct_illum[launch_index],1);
	  if (view_mode == 2)
	  {
	  output_buffer[launch_index] = make_float4(
	  indirect_illum_full,1);
	  }
	  if (view_mode == 3)
	  {
	  output_buffer[launch_index] = make_float4(
	  indirect_illum_combined,1);
	  }
	  if (view_mode == 4)
	  {
	  output_buffer[launch_index] = make_float4(indirect_illum_full,1);
	  }
	  if (view_mode == 5)
	  {
	  output_buffer[launch_index] = make_float4(
	  indirect_illum_combined,1);
	  }
	  
	  
	  }
	  */
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

  if(!visible[launch_index])
  {
    indirect_illum[launch_index] = make_float3(0);
    return;
  }

  if (gt_pass == 0)
	  indirect_illum[launch_index] = make_float3(0);

  int samp_this_pass = gt_samples_per_pass;
  if (gt_pass == gt_total_pass - 1)
  {
    samp_this_pass = total_gt_samples-samples_sofar;
  }

  float3 first_hit = world_loc[launch_index];
  float3 normal = n[launch_index];
  float3 Kd = Kd_image[launch_index];

  size_t2 out_buf = output_buffer.size();

  //ignore buckets, and place all results in the first bucket multiplied
  //by number of buckets used in non-gt code so we can use the same code
  float3 incoming_indirect_diffuse = make_float3(0);
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
    float3 prev_dir = normalize(first_hit-eye);
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
      float3 incoming_light = prd.incoming_diffuse_light * prd.Kd;
      incoming_diffuse += incoming_light;
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
    }
    incoming_indirect_diffuse += incoming_diffuse;

  }

  indirect_illum[launch_index] += incoming_indirect_diffuse
    /total_gt_samples;

}
