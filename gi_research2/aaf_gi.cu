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
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
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
  float3 incoming_direct_light;
  float3 norm;
  float3 Kd;
  float3 Ks;
  float z_dist;
  bool hit;
};
rtBuffer<float3, 2>               direct_illum;
rtBuffer<float3, 2>               indirect_illum;
rtBuffer<float3, 2>               indirect_illum_filter1d;
rtBuffer<float3, 2>               Kd_image;
rtBuffer<float3, 2>               Ks_image;
rtBuffer<float, 2>                target_indirect_spp;
rtBuffer<float2, 2>               z_dist;
rtBuffer<float2, 2>               z_dist_filter1d;
rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<float, 2>                depth;
rtBuffer<char, 2>                 visible;
rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(uint,  direct_ray_type, , );
rtDeclareVariable(uint,  num_buckets, , );
rtDeclareVariable(int,  z_filter_radius, , );
rtDeclareVariable(float,  vfov, , );
rtDeclareVariable(uint, max_spb_pass, , );
rtDeclareVariable(uint, indirect_ray_depth, , );
rtDeclareVariable(int, pixel_radius, , );
rtDeclareVariable(uint, use_textures, , );

rtDeclareVariable(uint, view_mode, , );
rtDeclareVariable(uint, view_bucket, , );
rtDeclareVariable(float, max_heatmap, , );

rtBuffer<float, 2>                 target_spb;

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtTextureSampler<float4, 2>   diffuse_map;  


//Filter functions
// Our Gaussian Filter, based on w_xf
//wxf= 1/b
__device__ __inline__ float gaussFilter(float distsq, float wxf)
{
  float sample = distsq*wxf*wxf;
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
    int i,
    int j,
    const size_t2& buf_size,
    unsigned int pass,
    unsigned int bucket)
{
  //const float dist_scale_threshold = 10.0f;
  const float z_thres = .1f;
  const float dist_threshold = 100.0f;
  const float angle_threshold = 10.f * M_PI/180.0f;

  if (i > 0 && i < buf_size.x && j > 0 && j < buf_size.y) {
    uint2 target_index = make_uint2(i,j);
    uint2 target_bucket_index = make_uint2(i,num_buckets*j+bucket);
    float3 target_indirect = indirect_illum[target_bucket_index];
    if (pass == 1)
      target_indirect = indirect_illum_filter1d[target_bucket_index];
    float target_zpmin = z_dist[target_bucket_index].x;
    float3 target_n = n[target_index];
    bool use_filt = visible[target_index];

    if (use_filt 
        && abs(acos(dot(target_n, cur_n))) < angle_threshold
        //&& abs((1./target_zpmin - 1./cur_zpmin)/(1./target_zpmin + 1./cur_zpmin)) < z_thres
        //&& abs((target_zpmin - cur_zpmin)/(target_zpmin + cur_zpmin)) < z_thres
        //&& (1/target_zpmin - 1/cur_zpmin) < dist_scale_threshold
       )
    {
      float3 target_loc = world_loc[target_index];
      float3 diff = cur_world_loc - target_loc;
      float euclidean_distsq = dot(diff,diff);
      float rDn = dot(diff, cur_n);
      float proj_distsq = euclidean_distsq - rDn*rDn;

      if (euclidean_distsq < (dist_threshold*dist_threshold))
      {
        float weight = gaussFilter(proj_distsq, 
            2.f*(1+50.*acos(dot(target_n,cur_n)))/cur_zpmin);

        blurred_indirect_sum += weight * target_indirect;
        sum_weight += weight;
      }
    }

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
  float3 cur_Kd =  Kd;
  if (use_textures)
  {
    float2 uv = make_float2(texcoord);
    cur_Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
  }
  prd_direct.Kd = cur_Kd;
  prd_direct.Ks = Ks;


  //lights
  unsigned int num_lights = lights.size();
  float3 direct = make_float3(0.0f);
  for(int i = 0; i < num_lights; ++i)
  {
    ParallelogramLight light = lights[i];
    float3 light_pos = light.corner;
    float Ldist = length(light_pos - hit_point);
    float3 L = normalize(light_pos - hit_point);
    float nDl = max(dot(ffnormal, L),0.f);
    float A = length(cross(light.v1, light.v2));
    float LnDl = dot( light.normal, L );
    float weight=nDl;// / (M_PIf*Ldist*Ldist);

    // cast shadow ray
    if ( nDl > 0.0f ) {
      PerRayData_pathtrace_shadow shadow_prd;
      shadow_prd.inShadow = false;
      Ray shadow_ray = make_Ray( hit_point, L, pathtrace_shadow_ray_type, 
          scene_epsilon, Ldist );
      rtTrace(top_object, shadow_ray, shadow_prd);

      if(!shadow_prd.inShadow){
        direct += make_float3(weight);
      }
    }

  }

  prd_direct.incoming_direct_light = direct;
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
      target_indirect_spp[bucket_index] = 0;
    }
    return;
  }
  visible[launch_index] = true;
  
  world_loc[launch_index] = dir_samp.world_loc;
  direct_illum[launch_index] = dir_samp.incoming_direct_light * dir_samp.Kd;
  n[launch_index] = dir_samp.norm;
  Kd_image[launch_index] = dir_samp.Kd;
  Ks_image[launch_index] = dir_samp.Ks;
  depth[launch_index] = dir_samp.z_dist;

  int initial_bucket_samples = 2;


  float3 n_u, n_v, n_w;
  createONB(dir_samp.norm, n_u, n_v, n_w);
  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x,
      frame_number);

  for(int bucket = 0; bucket < num_buckets; ++bucket)
  {
    uint2 bucket_index = make_uint2(base_bucket_index.x,
        base_bucket_index.y+bucket);
    float bucket_zmin_dist = 1000000000000.f;
    float bucket_zmax_dist = 0.f;
    for(int samp = 0; samp < initial_bucket_samples; ++samp)
    {
      float3 sample_dir;
      float2 rand_samp = make_float2(rnd(seed),rnd(seed));
      //vary theta component of sampling to sample split hemisphere
      rand_samp.x = (bucket + rand_samp.x)/num_buckets;
      //dont accept "grazing" angles
      rand_samp.y *= 0.95;
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
    indirect_illum[launch_index] = make_float3(0);
    return;
  }
  
  //calculate SPP
  float2 cur_zd = z_dist[launch_index]; //x: zmin, y:zmax
  size_t2 screen_size = output_buffer.size();
  float proj_dist = 2./screen_size.y * depth[screen_index] 
    * tan(vfov/2.*M_PI/180.);
  float wvmax = 2; //Constant for now (diffuse)
  float alpha = 1;
  float spp_term1 = proj_dist * wvmax/cur_zd.x + alpha;
  float spp_term2 = 1+cur_zd.y/cur_zd.x;

  float spp = spp_term1*spp_term1 * wvmax*wvmax * spp_term2*spp_term2;
  
  //account for split buckets
  spp /= num_buckets;

  target_spb[launch_index] = spp;

  spp = max(min(spp, (float)max_spb_pass),1.f);


  int spp_int = (int) spp;

  float3 first_hit = world_loc[screen_index];
  float3 normal = n[screen_index];
  float3 Kd = Kd_image[screen_index];

  //sample this hemisphere according to our spp
  float3 incoming_indirect;
  unsigned int seed = tea<16>(bucket.x*launch_index.y+launch_index.x,
      frame_number); //TODO :verify
  for (int samp = 0; samp < spp_int; ++samp)
  {
    PerRayData_direct prd;
    float3 ray_origin = first_hit;
    float3 ray_n = normal;
    float3 rn_u, rn_v, rn_w;
    float3 sample_dir;
    float3 sample_color = make_float3(0);
    for (int depth = 0; depth < indirect_ray_depth; ++depth)
    {
      prd.hit = false;
      createONB(ray_n, rn_u, rn_v, rn_w);
      float2 rand_samp = make_float2(rnd(seed), rnd(seed));
      rand_samp.x = (cur_bucket + rand_samp.x)/num_buckets;
      sampleUnitHemisphere(rand_samp, rn_u, rn_v, rn_w, sample_dir);
      Ray ray = make_Ray(ray_origin, sample_dir,
          direct_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if (!prd.hit)
        break;
      sample_color += prd.incoming_direct_light * prd.Kd;
      ray_origin = prd.world_loc;
      ray_n = prd.norm;
    }
    incoming_indirect += sample_color;
  }
  incoming_indirect /= (float)spp_int;
  
  indirect_illum[launch_index] = incoming_indirect;

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

  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];

  for (int i = -pixel_radius; i < pixel_radius; ++i)
  {
    if (visible[screen_index])
    {
      indirectFilter(blurred_indirect_sum, sum_weight,
          cur_world_loc, cur_n, cur_zmin,
          screen_index.x+i, screen_index.y,
          buf_size, 0, cur_bucket);
    }
  }

  if (sum_weight > 0.0001f)
    indirect_illum_filter1d[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum_filter1d[launch_index] = indirect_illum[launch_index];
}
RT_PROGRAM void indirect_filter_second_pass()
{
  uint2 screen_index = make_uint2(launch_index.x,
      launch_index.y/num_buckets);
  uint cur_bucket = launch_index.y%num_buckets;

  float cur_zmin = z_dist[launch_index].x;
  size_t2 buf_size = indirect_illum.size();
  float3 blurred_indirect_sum = make_float3(0.f);
  float sum_weight = 0.f;

  float3 cur_world_loc = world_loc[screen_index];
  float3 cur_n = n[screen_index];

  for (int i = -pixel_radius; i < pixel_radius; ++i)
  {
    if (visible[screen_index])
    {
      indirectFilter(blurred_indirect_sum, sum_weight,
          cur_world_loc, cur_n, cur_zmin,
          screen_index.x, screen_index.y+i,
          buf_size, 1, cur_bucket);
    }
  }

  if (sum_weight > 0.0001f)
    indirect_illum[launch_index] = blurred_indirect_sum/sum_weight;
  else
    indirect_illum[launch_index] = indirect_illum_filter1d[launch_index];


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
  for (int i = 0; i < num_buckets; ++i)
  {
    indirect_illum_combined += indirect_illum[make_uint2(launch_index.x, 
        launch_index.y*num_buckets+i)];
  }

  indirect_illum_combined *= Kd_image[launch_index]/num_buckets;
  output_buffer[launch_index] = make_float4(
      direct_illum[launch_index] + indirect_illum_combined,1);

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
            indirect_illum[target_bucket_index] * Kd_image[launch_index],1);
      else
        output_buffer[launch_index] = make_float4(indirect_illum_combined,1);
    }
    if (view_mode == 3)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(
            indirect_illum[target_bucket_index],1);
      else
        output_buffer[launch_index] = make_float4(
            indirect_illum_combined/Kd_image[launch_index],1);
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
    if (view_mode == 4)
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
    if (view_mode == 5)
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
    if (view_mode == 6)
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
    if (view_mode == 7)
    {
      if (view_separated_bucket)
        output_buffer[launch_index] = make_float4(heatMap(
              min(target_spb[target_bucket_index],(float)max_spb_pass)
              /max_heatmap));
      else
      {
        float combined_spp;
        for (int i = 0; i < num_buckets; ++i)
        {
          combined_spp += target_spb[make_uint2(launch_index.x,
                launch_index.y*num_buckets+i)];
        }
        output_buffer[launch_index] = make_float4(heatMap(
              min(combined_spp,(float)max_spb_pass)
              /(max_heatmap*num_buckets)));
      }

    }
  }

}
