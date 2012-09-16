/*
 * arealight.h
 * Area Light Filtering
 * Adapted from NVIDIA OptiX Tutorial
 * Brandon Wang, Soham Mehta
 */

#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"
#include "random.h"

using namespace optix;

#define FLT_MAX         1e30;

__device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

__device__ __inline__ float step( float min, float value )
{
  return value<min?0:1;
}

__device__ __inline__ float3 mix( float3 a, float3 b, float x )
{
  return a*(1-x) + b*x;
}

__device__ __inline__ float3 schlick( float nDi, const float3& rgb )
{
  float r = fresnel_schlick(nDi, 5, rgb.x, 1);
  float g = fresnel_schlick(nDi, 5, rgb.y, 1);
  float b = fresnel_schlick(nDi, 5, rgb.z, 1);
  return make_float3(r, g, b);
}

__device__ __inline__ uchar4 make_color(const float3& c)
{
  return make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
      static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
      static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
      255u);                                                 /* A */
}

struct PerRayData_radiance
{
  // Things to specify when creating the ray
  bool first_pass;
  int sqrt_num_samples;

  // Things to specify and will get updated
  float unavg_vis;
  float vis_weight_tot;
  
  // Things that will be generated automatically
  bool hit_shadow;
  bool use_filter_n;
  float s1;
  float s2;
  float dist_to_light;

  // Things that should only be generated on first pass
  bool hit;
  float t_hit;
  float3 world_loc;
  float3 n;
  float3 brdf;
  int obj_id;
};

struct PerRayData_shadow
{
  bool hit;
  float3 attenuation;
  float distance_min;
  float distance_max;
};

