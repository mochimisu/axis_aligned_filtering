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
#define MIN_Z_MIN 15.0f

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
rtBuffer<float3, 2>               indirect_illum_filter1d_out;
rtBuffer<float3, 2>               indirect_illum_filter1d_in;
//specular buffers
rtBuffer<float3, 2>               indirect_illum_spec;
rtBuffer<float3, 2>               indirect_illum_spec_filter1d_out;
rtBuffer<float3, 2>               indirect_illum_spec_filter1d_in;

rtBuffer<float3, 2>               Kd_image;
rtBuffer<float3, 2>               Ks_image;
rtBuffer<float, 2>                phong_exp_image;
rtBuffer<filter_info, 2>          filter_info_in;
rtBuffer<filter_info, 2>          filter_info_out;
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

  filter_info target_finfo = filter_info_in[target_index];

  float target_zpmin = target_finfo.zmin;
  float3 target_n = target_finfo.n;
  bool use_filt = target_finfo.valid;

  if (use_filt
      && abs(acos(dot(target_n, cur_n))) < angle_threshold
     )
  {
    float3 target_loc = target_finfo.world_loc;
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
	  filter_info cur_finfo = filter_info_in[target_index];

	  if(!cur_finfo.valid)
		  return;

    float3 target_loc = cur_finfo.world_loc;
    float3 diff = cur_world_loc - target_loc;
    float euclidean_distsq = dot(diff,diff);
    float rDn = dot(diff, cur_n);
    float proj_distsq = euclidean_distsq - rDn*rDn;
    float3 target_n = cur_finfo.n;

    float diff_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
        OHMAX);
    float3 target_indirect = cur_finfo.indirect_diffuse;

#ifdef FILTER_SPECULAR
	float spec_weight = filterWeight(proj_distsq, target_n, cur_n, cur_zpmin,
	cur_spec_wvmax);
	float3 target_indirect_spec = cur_finfo.indirect_specular;
#endif
    if (pass == 1)
    {
      target_indirect = indirect_illum_filter1d_in[target_index];
#ifdef FILTER_SPECULAR
      target_indirect_spec = indirect_illum_spec_filter1d_in[target_index];
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



RT_PROGRAM void sample_aaf()
{
	size_t2 screen = direct_illum.size();

	filter_info finfo;

	//direct sample
	float3 ray_origin = eye;
	float2 d = make_float2(launch_index)/make_float2(screen) * 2.f - 1.f;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	indirect_illum[launch_index] = make_float3(0);
	indirect_illum_spec[launch_index] = make_float3(0);

	PerRayData_direct dir_samp;
	dir_samp.hit = false;
	Ray ray = make_Ray(ray_origin, ray_direction, direct_ray_type, 
		scene_epsilon, RT_DEFAULT_MAX);
	rtTrace(top_object, ray, dir_samp);

	float2 cur_zdist = make_float2(10000000000.f,scene_epsilon);
	float cur_depth;

	if (!dir_samp.hit) {
		direct_illum[launch_index] = make_float3(0.34f,0.55f,0.85f);
		target_spb_theoretical[launch_index] = 0;
		target_spb[launch_index] = 0;
		target_spb_spec_theoretical[launch_index] = 0;
		target_spb_spec[launch_index] = 0;
		finfo.valid = false;
		return;
	}
	direct_illum[launch_index] = dir_samp.incoming_diffuse_light * dir_samp.Kd
		+ dir_samp.incoming_specular_light * dir_samp.Ks;
	finfo.valid = true;
	finfo.world_loc = dir_samp.world_loc;
	finfo.indirect_diffuse = make_float3(0);
	finfo.indirect_specular = make_float3(0);
	finfo.n = dir_samp.norm;
	finfo.zmin = 1000000000.f;
	finfo.spec_wvmax = glossy_blim(dir_samp.phong_exp);

	Kd_image[launch_index] = dir_samp.Kd;
	Ks_image[launch_index] = dir_samp.Ks;

	cur_depth = dir_samp.z_dist;


	int initial_bucket_samples_sqrt = 4;
	int initial_bucket_samples = initial_bucket_samples_sqrt
		* initial_bucket_samples_sqrt;


	float3 n_u, n_v, n_w;
	createONB(dir_samp.norm, n_u, n_v, n_w);
	unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
		frame_number);

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
			cur_zdist.x = clamp(indir_samp.z_dist, MIN_Z_MIN, cur_zdist.x);
			cur_zdist.y = max(cur_zdist.y, indir_samp.z_dist);
		}
	}

	// spp
	float proj_dist = 2.f/screen.y * cur_depth 
		* tan(vfov/2.f*M_PI/180.f);
	finfo.proj_dist = proj_dist;
	float alpha = 1.f;
	float spp_term1 = OHMAX * spp_mu * proj_dist/cur_zdist.x + alpha;
	float spp_term2 = 1.f+spp_mu*cur_zdist.y/cur_zdist.x;
	
	float spp = imp_samp_scale_diffuse
		*spp_term1*spp_term1
		* spp_term2*spp_term2 * OHMAX * OHMAX;
	
	
	float cur_spec_wvmax = finfo.spec_wvmax;
	float spp_spec_term1 = proj_dist * cur_spec_wvmax/cur_zdist.x + alpha;
	
	float spec_spp = imp_samp_scale_specular
		* spp_spec_term1 * spp_spec_term1 
		* cur_spec_wvmax*cur_spec_wvmax
		* spp_term2*spp_term2;
	
	target_spb_theoretical[launch_index] = spp;
	
	
	
	float Kd_mag = length(dir_samp.Kd);
	float Ks_mag = length(dir_samp.Ks);
	float Kd_Ks_ratio = Kd_mag/(Kd_mag+Ks_mag);
	
	spp = clampVal(spp * Kd_Ks_ratio, 0.f, (float)spp_mu*max_spb_pass);
	spp = 100;
	spec_spp = clampVal(spec_spp* (1.f-Kd_Ks_ratio), 
	0.f, (float) spp_mu*max_spb_spec_pass);

	float spp_sqrt = sqrt(spp);
	int spp_sqrt_int = (int) ceil(spp_sqrt);
	int spp_int = spp_sqrt_int * spp_sqrt_int;
	target_spb[launch_index] = spp_int;

	//output_buffer[launch_index] = make_float4(spp_int/100.f);
	
#ifdef SAMPLE_SPECULAR
	float spp_spec_sqrt = sqrt(spec_spp);
	int spp_spec_sqrt_int = (int) ceil(spp_spec_sqrt);
	int spp_spec_int = spp_spec_sqrt_int * spp_spec_sqrt_int;
	target_spb_spec_theoretical[launch_index] = spec_spp;
	target_spb_spec[launch_index] = spp_spec_int;
#else
	target_spb_spec_theoretical[launch_index] = 0;
	target_spb_spec[launch_index] = 0;
#endif

	//indirect sampling
	float3 first_hit = dir_samp.world_loc;
	float3 normal = dir_samp.norm;

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
		float prev_phong_exp = dir_samp.phong_exp;
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
		finfo.indirect_diffuse += sample_diffuse_color;
		finfo.indirect_specular += sample_specular_color;
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
		float prev_phong_exp = dir_samp.phong_exp;
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

			sampleUnitHemispherePower(rand_samp, rns_u, rns_v, rns_w, dir_samp.phong_exp,
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
		finfo.indirect_diffuse += sample_diffuse_color;
		finfo.indirect_specular += sample_specular_color;
	}
	finfo.indirect_specular /= (float)spp_int+spp_spec_int;
#else
	finfo.indirect_specular /= (float)spp_int;
#endif
	//incoming_diff_indirect /= (float)spp_int+spp_spec_int;
	finfo.indirect_diffuse /= (float)spp_int;
	//incoming_spec_indirect /= (float)spp_int;

	finfo.zmin = cur_zdist.x;


	filter_info_out[launch_index] = finfo;
}

RT_PROGRAM void indirect_filter_first_pass()
{
	size_t2 buf_size = filter_info_in.size();
	filter_info cur_finfo = filter_info_in[launch_index];
	float cur_zmin = cur_finfo.zmin;

	float3 blurred_indirect_sum = make_float3(0.f);
	float sum_weight = 0.f;

	float3 blurred_indirect_spec_sum = make_float3(0.f);
	float sum_weight_spec = 0.f;

	float cur_spec_wvmax = cur_finfo.spec_wvmax;

	float3 cur_world_loc = cur_finfo.world_loc;
	float3 cur_n = cur_finfo.n;

	float proj_dist = cur_finfo.proj_dist;
	int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);
  

	if (cur_finfo.valid)
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
			indirect_illum_filter1d_out[launch_index] = blurred_indirect_sum/sum_weight;
		else
			indirect_illum_filter1d_out[launch_index] = cur_finfo.indirect_diffuse;
#ifdef FILTER_SPECULAR
		if (sum_weight_spec > 0.0001f)
			indirect_illum_spec_filter1d_out[launch_index] = blurred_indirect_spec_sum
			/sum_weight_spec;
		else
			indirect_illum_spec_filter1d_out[launch_index] = cur_finfo.indirect_specular;
#endif

}
RT_PROGRAM void indirect_filter_second_pass()
{
  
	size_t2 buf_size = filter_info_in.size();
	filter_info cur_finfo = filter_info_in[launch_index];
	float cur_zmin = cur_finfo.zmin;

	float3 blurred_indirect_sum = make_float3(0.f);
	float sum_weight = 0.f;

	float3 blurred_indirect_spec_sum = make_float3(0.f);
	float sum_weight_spec = 0.f;

	float cur_spec_wvmax = cur_finfo.spec_wvmax;

	float3 cur_world_loc = cur_finfo.world_loc;
	float3 cur_n = cur_finfo.n;

	float proj_dist = cur_finfo.proj_dist;
	int radius = clampVal( 2.f*cur_zmin/(spp_mu*OHMAX*proj_dist) , 1.f, MAX_FILT_RADIUS);

  
  if (cur_finfo.valid)
 
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
  else
	  indirect_illum[launch_index] = make_float3(0);

  if (sum_weight > 0.0001f)
    indirect_illum[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum[launch_index] = indirect_illum_filter1d_in[launch_index];
#ifdef FILTER_SPECULAR
  if (sum_weight_spec > 0.0001f)
    indirect_illum_spec[launch_index] = blurred_indirect_spec_sum
      /sum_weight_spec;
  else
    indirect_illum_spec[launch_index] = 
      indirect_illum_spec_filter1d_in[launch_index];
#endif

}


RT_PROGRAM void display()
{
	//output_buffer[launch_index] = make_float4(direct_illum[launch_index]);

	output_buffer[launch_index] = make_float4(
		direct_illum[launch_index]
	  + indirect_illum[launch_index] * Kd_image[launch_index]
	  + indirect_illum_spec[launch_index] * Ks_image[launch_index]
	  ,1.f);
	  /*
	  //output_buffer[launch_index] = make_float4(indirect_illum[launch_index]);
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
  }*/

}
