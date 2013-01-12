#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "aaf_gi.h"
#include "random.h"

#define MULTI_BOUNCE
#define SAMPLE_SPECULAR
#define FILTER_SPECULAR

#define MAX_FILT_RADIUS 50.f
#define OHMAX 2.8f
#define MIN_Z_MIN 0.1f

using namespace optix;

#define clampVal(a,b,c) max(b,min(c,a))

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
rtDeclareVariable(int,  z_filter_radius, , );
rtDeclareVariable(float,  vfov, , );
rtDeclareVariable(uint, max_spb_pass, , );
rtDeclareVariable(uint, max_spb_spec_pass, , );
rtDeclareVariable(uint, indirect_ray_depth, , );
rtDeclareVariable(int, pixel_radius, , );
rtDeclareVariable(uint, use_textures, , );
rtDeclareVariable(float, spp_mu, , );
rtDeclareVariable(float, imp_samp_scale_diffuse, ,);


rtDeclareVariable(uint, view_mode, , );
rtDeclareVariable(float, max_heatmap, , );

rtBuffer<float, 2>                 target_spb_theoretical;
rtBuffer<float, 2>                 target_spb;

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2>   diffuse_map;  
rtTextureSampler<float4, 2>   specular_map;  

//prefilter pass
//x: number of rejected pixels
//y: number of possible filtered pixels
rtBuffer<uint2, 2>                prefilter_rejected_filter1d;
rtBuffer<uint2, 2>                prefilter_rejected;

rtDeclareVariable(float, imp_samp_scale_specular, ,);

rtBuffer<float, 2>                 target_spb_spec_theoretical;
rtBuffer<float, 2>                 target_spb_spec;


//omegaxf and sampling functions
__device__ __inline__ float glossy_blim(float m){
	return 3.5 + 0.2*m;
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
  if (sample > 2.) {
    return 0.0;
  }

  return exp(-2.*sample);
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

__device__ __inline__ float filterWeight(
    float proj_distsq, const float3& target_n, const float3& cur_n,
    float cur_zpmin, float wvmax)
{
	float wxf = wvmax*(1.f+50.f*acos(dot(target_n,cur_n)))/cur_zpmin;	
	float sample = proj_distsq*wxf*wxf*spp_mu*spp_mu;
	if (sample > 2.f) {
		return 0.0;
	}
	return exp(-2.f*sample);
}


__device__ __inline__ void indirectFilter( 
    float3& blurred_indirect_sum,
    float& sum_weight,
#ifdef FILTER_SPECULAR
    float3& blurred_indirect_spec_sum,
    float& sum_weight_spec,
    const float cur_spec_wvmax,
#endif
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

    float diff_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
        OHMAX);
    float3 target_indirect = indirect_illum[target_index];

#ifdef FILTER_SPECULAR
	float spec_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
	cur_spec_wvmax);
	float3 target_indirect_spec = indirect_illum_spec[target_index];
#endif
    if (pass == 1)
    {
      target_indirect = indirect_illum_filter1d[target_index];
#ifdef FILTER_SPECULAR
      target_indirect_spec = indirect_illum_spec_filter1d[target_index];
#endif
    }

    blurred_indirect_sum += diff_weight * target_indirect;
    sum_weight += diff_weight;

#ifdef FILTER_SPECULAR
    blurred_indirect_spec_sum += spec_weight * target_indirect_spec;
    sum_weight_spec += spec_weight;
#endif
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

  //initialize some buffers
  prefilter_rejected[launch_index] = make_uint2(0,1);

    indirect_illum[launch_index] = make_float3(0);
    indirect_illum_spec[launch_index] = make_float3(0);

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
  direct_illum[launch_index] = dir_samp.incoming_diffuse_light * dir_samp.Kd
    + dir_samp.incoming_specular_light * dir_samp.Ks;
  n[launch_index] = dir_samp.norm;
  Kd_image[launch_index] = dir_samp.Kd;
  Ks_image[launch_index] = dir_samp.Ks;
  depth[launch_index] = dir_samp.z_dist;

  //use R with light or R from camera?
  float cur_spec_wvmax = glossy_blim(dir_samp.phong_exp);
  phong_exp_image[launch_index] = dir_samp.phong_exp;
  spec_wvmax[launch_index] = cur_spec_wvmax;


  int initial_bucket_samples_sqrt = 4;
  int initial_bucket_samples = initial_bucket_samples_sqrt
    * initial_bucket_samples_sqrt;


  float3 n_u, n_v, n_w;
  createONB(dir_samp.norm, n_u, n_v, n_w);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
	  frame_number);

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
		  bucket_zmin_dist = clamp(indir_samp.z_dist, MIN_Z_MIN, bucket_zmin_dist);
		  bucket_zmax_dist = max(bucket_zmax_dist, indir_samp.z_dist);
	  }
  }
  z_dist[launch_index] = make_float2(bucket_zmin_dist, bucket_zmax_dist);
}

RT_PROGRAM void sample_indirect()
{

  if(!visible[launch_index])
  {
    target_spb_theoretical[launch_index] = 0;
    target_spb[launch_index] = 0;
    target_spb_spec_theoretical[launch_index] = 0;
    target_spb_spec[launch_index] = 0;
    indirect_illum[launch_index] = make_float3(0);
    return;
  }

  float3 Kd = Kd_image[launch_index];
  float3 Ks = Ks_image[launch_index];
  float cur_phong_exp = phong_exp_image[launch_index];
  
  //calculate SPP
  float2 cur_zd = z_dist[launch_index]; //x: zmin, y:zmax
  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] 
    * tan(vfov/2.*M_PI/180.);
  float alpha = 1.f;
  float spp_term1 = OHMAX * spp_mu * proj_dist/cur_zd.x + alpha;
  float spp_term2 = 1.f+spp_mu*cur_zd.y/cur_zd.x;

  float spp = imp_samp_scale_diffuse
	*spp_term1*spp_term1
    * spp_term2*spp_term2 * OHMAX * OHMAX;




  float cur_spec_wvmax = spec_wvmax[launch_index];
  float spp_spec_term1 = proj_dist * cur_spec_wvmax/cur_zd.x + alpha;

  float spec_spp = imp_samp_scale_specular
	* spp_spec_term1 * spp_spec_term1 
    * cur_spec_wvmax*cur_spec_wvmax
    * spp_term2*spp_term2;

  target_spb_theoretical[launch_index] = spp;
  target_spb_spec_theoretical[launch_index] = spec_spp;

  uint2 pf_rej = prefilter_rejected[launch_index];
  
  float rej_scale = (1.f + (float)pf_rej.x/pf_rej.y);

  float Kd_mag = length(Kd);
  float Ks_mag = length(Ks);
  float Kd_Ks_ratio = Kd_mag/(Kd_mag+Ks_mag);

  spp = max(
      min(spp / (1.f-(float)pf_rej.x/pf_rej.y), 
        (float)max_spb_pass*spp_mu),
      1.f) * Kd_Ks_ratio;
  spec_spp = max( min(spp, (float)max_spb_spec_pass*spp_mu), 1.f)
    *(1.f-Kd_Ks_ratio); 
  //TODO: distribute samples according to kd to ks ratio, account for prefilt

  float spp_sqrt = sqrt(spp);
  int spp_sqrt_int = (int) ceil(spp_sqrt);
  int spp_int = spp_sqrt_int * spp_sqrt_int;
  target_spb[launch_index] = spp_int;

#ifdef SAMPLE_SPECULAR
  float spp_spec_sqrt = sqrt(spec_spp);
  int spp_spec_sqrt_int = (int) ceil(spp_spec_sqrt);
  int spp_spec_int = spp_spec_sqrt_int * spp_spec_sqrt_int;
  target_spb_spec[launch_index] = spp_spec_int;
#endif

  float3 first_hit = world_loc[launch_index];
  float3 normal = n[launch_index];

  //diffuse
  //sample this hemisphere with cosine (aligned to normal) weighting
  float3 incoming_diff_indirect = make_float3(0);
  float3 incoming_spec_indirect = make_float3(0);
  size_t2 screen = direct_illum.size();
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
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
#ifdef MULTI_BOUNCE
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
#endif
      prd.hit = false;
      createONB(ray_n, rn_u, rn_v, rn_w);
      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%spp_sqrt_int + rand_samp.x)/spp_sqrt_int;
      rand_samp.y = (((int)samp/spp_sqrt_int) + rand_samp.y)/spp_sqrt_int;

      sampleUnitHemisphere(rand_samp, rn_u, rn_v, rn_w, sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
	  if (!prd.hit)
#ifdef MULTI_BOUNCE
		  break;
#else
		  continue;
#endif
      float3 R = normalize(2*ray_n*dot(ray_n, sample_dir)-sample_dir);
      float nDr = max(dot(-prev_dir, R), 0.f);
      float3 incoming_light = prd.incoming_specular_light * prd.Ks 
        + prd.incoming_diffuse_light * prd.Kd;

      sample_diffuse_color += incoming_light * prev_Kd;
      sample_specular_color += incoming_light*pow(nDr, prev_phong_exp) 
		  * prev_Ks;
#ifdef MULTI_BOUNCE
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
      prev_phong_exp = prd.phong_exp;
	  prev_Kd = prd.Kd;
	  prev_Ks = prd.Ks;
    }
#endif
    incoming_diff_indirect += sample_diffuse_color;
    incoming_spec_indirect += sample_specular_color;
  }

  //specular
  //sample this hemisphere with cos^n (aligned to reflected angle) weighting
#ifdef SAMPLE_SPECULAR
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
#ifdef MULTI_BOUNCE
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
#endif
      prd.hit = false;

      prev_dir = normalize(prev_dir);
      float3 perf_refl = normalize(prev_dir - 2*ray_n*dot(ray_n, prev_dir));
      createONB(perf_refl, rns_u, rns_v, rns_w);

      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      //stratify x,y
      rand_samp.x = (samp%spp_spec_sqrt_int + rand_samp.x)/spp_spec_sqrt_int;
      rand_samp.y = (((int)samp/spp_spec_sqrt_int) + rand_samp.y)
        /spp_spec_sqrt_int;

      sampleUnitHemispherePower(rand_samp, rns_u, rns_v, rns_w, cur_phong_exp,
          sample_dir);

      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
	  if (!prd.hit)
#ifdef MULTI_BOUNCE
	  break;
#else
	  continue;
#endif

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

#ifdef MULTI_BOUNCE
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
      prev_dir = sample_dir;
      prev_phong_exp = prd.phong_exp;
	  prev_Ks = prd.Ks;
	  prev_Kd = prd.Kd;
    }
#endif
    incoming_diff_indirect += sample_diffuse_color;
    incoming_spec_indirect += sample_specular_color;
  }
  incoming_spec_indirect /= (float)spp_int+spp_spec_int;
#else
  incoming_spec_indirect /= (float)spp_int;
#endif
  //incoming_diff_indirect /= (float)spp_int+spp_spec_int;
  incoming_diff_indirect /= (float)spp_int;
  //incoming_spec_indirect /= (float)spp_int;
  
  indirect_illum[launch_index] = incoming_diff_indirect;
  indirect_illum_spec[launch_index] = incoming_spec_indirect;

}
RT_PROGRAM void indirect_filter_first_pass()
{
  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 blurred_indirect_spec_sum = make_float3(0.f);
  float sum_weight_spec = 0.f;

  float cur_spec_wvmax = spec_wvmax[launch_index];

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] * tan(vfov/2.*M_PI/180.);
  int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);
  
  //z_dist[launch_index].y = radius;	//SCREEN SPACE RADIUS

  if (visible[launch_index])
    for (int i = -radius; i < radius; ++i)
    {
      uint2 target_index = make_uint2(launch_index.x+i, launch_index.y);
      indirectFilter(blurred_indirect_sum, sum_weight,
#ifdef FILTER_SPECULAR
          blurred_indirect_spec_sum, sum_weight_spec, cur_spec_wvmax,
#endif
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 0);
    }

  if (sum_weight > 0.0001f)
    indirect_illum_filter1d[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum_filter1d[launch_index] = indirect_illum[launch_index];
#ifdef FILTER_SPECULAR
  if (sum_weight_spec > 0.0001f)
    indirect_illum_spec_filter1d[launch_index] = blurred_indirect_spec_sum
      /sum_weight_spec;
  else
    indirect_illum_spec_filter1d[launch_index] = 
      indirect_illum_spec[launch_index];
#endif
}
RT_PROGRAM void indirect_filter_second_pass()
{
  
  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 blurred_indirect_spec_sum = make_float3(0.f);
  float sum_weight_spec = 0.f;

  float cur_spec_wvmax = spec_wvmax[launch_index];


  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];
  
  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] * tan(vfov/2.*M_PI/180.);
  int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);
  
  //z_dist[launch_index].y = radius;	//SCREEN SPACE RADIUS
  //radius = 10u;

  if (visible[launch_index])
    for (int i = -radius; i < radius; ++i)
	  {
      uint2 target_index = make_uint2(launch_index.x, launch_index.y+i);
      indirectFilter(blurred_indirect_sum, sum_weight,
#ifdef FILTER_SPECULAR
          blurred_indirect_spec_sum, sum_weight_spec, cur_spec_wvmax,
#endif
          cur_world_loc, cur_n, cur_zmin,
          target_index, buf_size, 1);
    }

  if (sum_weight > 0.0001f)
    indirect_illum[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum[launch_index] = indirect_illum_filter1d[launch_index];
#ifdef FILTER_SPECULAR
  if (sum_weight_spec > 0.0001f)
    indirect_illum_spec[launch_index] = blurred_indirect_spec_sum
      /sum_weight_spec;
  else
    indirect_illum_spec[launch_index] = 
      indirect_illum_spec_filter1d[launch_index];
#endif

}
RT_PROGRAM void indirect_prefilter_first_pass()
{
  
  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();

  float3 cur_world_loc = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  uint2 cur_prefilter_rej = make_uint2(0,0);

  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] * tan(vfov/2.*M_PI/180.);
  int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);
  
  //z_dist[launch_index].y = radius;	//SCREEN SPACE RADIUS

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

  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[launch_index] * tan(vfov/2.*M_PI/180.);
  int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);
  
  //z_dist[launch_index].y = radius;	//SCREEN SPACE RADIUS

  if (visible[launch_index])
    for (int i = -radius; i < radius; ++i)
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
  float3 indirect_illum_spec_combined = indirect_illum_spec[
	  make_uint2(launch_index.x, launch_index.y)];

  float3 indirect_illum_full = indirect_illum_combined * Kd_image[launch_index];
  float3 indirect_illum_spec_full = indirect_illum_spec_combined 
    * Ks_image[launch_index];
  output_buffer[launch_index] = make_float4(
      direct_illum[launch_index] 
      + indirect_illum_spec_full + indirect_illum_full,1);
  //other view modes
  if (view_mode)
  {
    if (view_mode == 1)
      output_buffer[launch_index] = make_float4(direct_illum[launch_index],1);
    if (view_mode == 2)
    {
        output_buffer[launch_index] = make_float4(
            indirect_illum_full+indirect_illum_spec_full,1);
    }
    if (view_mode == 3)
    {
        output_buffer[launch_index] = make_float4(
            indirect_illum_combined+indirect_illum_spec_combined,1);
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

    if (view_mode == 6)
    {
        output_buffer[launch_index] = make_float4(
            indirect_illum_spec_full,1);
    }
    if (view_mode == 7)
    {
        output_buffer[launch_index] = make_float4(
            indirect_illum_spec_combined,1);
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
    float prev_phong_exp = phong_exp_image[launch_index];
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

  indirect_illum[launch_index] += incoming_indirect_diffuse
    /total_gt_samples;
  indirect_illum_spec[launch_index] += incoming_indirect_specular
    /total_gt_samples;

}
