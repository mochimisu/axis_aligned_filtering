/*
* arealight.cu
* Area Light Filtering
* Adapted from NVIDIA OptiX Tutorial
* Brandon Wang, Soham Mehta
*/


/*

   todo
   2 passes
   limit spp
   */

#include "filtergi.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

rtDeclareVariable(PerRayData_direct, prd_direct, rtPayload, );
rtDeclareVariable(PerRayData_indirect,   prd_indirect,   rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, direct_ray_type, , );
rtDeclareVariable(unsigned int, indirect_ray_type , , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );

rtDeclareVariable(unsigned int,        max_spp, , );

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

// sample hemisphere with cosine density
__device__ __inline__ void sampleUnitHemisphere( const optix::float2& sample,
  const optix::float3& U,
  const optix::float3& V,
  const optix::float3& W,
  optix::float3& point )
{
sampleUnitHemispherePower(sample,U,V,W,1,point);
/*
  using namespace optix;

  float phi = 2.0f * M_PIf*sample.x;
  float r = sqrt( sample.y );
  float x = r * cos(phi);
  float y = r * sin(phi);
  float z = 1.0f - x*x -y*y;
  z = z > 0.0f ? sqrt(z) : 0.0f;

  point = x*U + y*V + z*W;
  */
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

  return exp(-sample);
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

//omega_v_max for phong brdf
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
rtBuffer<float3, 2>               indirect_illum_sep;
rtBuffer<float2, 2>                z_perp;
rtBuffer<float3, 2>               indirect_illum_blur1d;
rtBuffer<float3, 2>               indirect_illum_filt;
rtBuffer<float3, 2>               indirect_illum_filt_int;
rtBuffer<float4, 2>               indirect_illum_accum;
rtBuffer<char, 2>                 use_filter;

rtBuffer<float3, 2>               world_loc;
rtBuffer<float3, 2>               n;
rtBuffer<int, 2>                  indirect_spp;
rtBuffer<int, 2>                  target_indirect_spp;

rtBuffer<float3, 2>               image_Kd;

rtTextureSampler<float4, 2>   diffuse_map;  

rtBuffer<float3, 2>               image_Ks;
rtBuffer<float, 2>               image_phong_exp;
rtBuffer<float, 2>                omega_v_max;

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


rtDeclareVariable(float,          max_disp_val, , );
rtDeclareVariable(float,          min_disp_val, , );

rtDeclareVariable(float,          spp_mu, , );

rtDeclareVariable(uint2,          image_dim, , );
rtDeclareVariable(float,          fov, , );

//Random direction buffer
rtBuffer<uint2, 2> indirect_rng_seeds;


RT_PROGRAM void pinhole_camera_initial_sample() {
  // Find direction to shoot ray in
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  PerRayData_direct prd;

  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);

  if (frame == 0)
  {
    for(int i = 0; i < 4; ++i)
    {
      uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+i);
      indirect_illum_accum[cur_bucket_index] = make_float4(0);
      indirect_illum_sep[cur_bucket_index] = make_float3(0,0,0);
      z_perp[cur_bucket_index] = make_float2(100000000,0);
      indirect_spp[cur_bucket_index] = 0;
      target_indirect_spp[cur_bucket_index] = 0;
    }

    use_filter[launch_index] = false;
    //indirect_spp[launch_index] = 0;
    //target_indirect_spp[launch_index] = 0;
    indirect_illum[launch_index] = make_float3(0);
    image_Kd[launch_index] = make_float3(0);
    image_Ks[launch_index] = make_float3(0);
    image_phong_exp[launch_index] = 0;
  }


  // Initialize the stuff we use in later passes

  optix::Ray ray(ray_origin, ray_direction, direct_ray_type, scene_epsilon);
  //prd.sqrt_num_samples = normal_rpp;
  prd.hit = false;

  rtTrace(top_object, ray, prd);
  
  use_filter[launch_index] = prd.hit;
  direct_illum[launch_index] = prd.color;
  //indirect_spp[launch_index] = prd.indirect_spp;

  if (!prd.hit)
    return;

  image_Kd[launch_index ] =prd.Kd;
  image_Ks[launch_index ] =prd.Ks;
  image_phong_exp[launch_index ] =prd.phong_exp;
  world_loc[launch_index] = prd.world_loc;
  //indirect_illum[launch_index] = prd.indirect;
  n[launch_index] = normalize(prd.n);

  //Shoot 16 rays initially for sampling for SPP
  uint2 seed = indirect_rng_seeds[launch_index];
  int init_ind_spp_sqrt = 4;
  float3 u,v,w;
  float3 sampleDir;
  createONB(prd.n, u,v,w);

  float2 sample = make_float2(0);
  
  int totbucket = 0;

  float zpmin[4] = { 100000, 100000, 100000, 100000 };
  float zpmax[4] = { 0, 0, 0, 0 };

  float4 cur_indirect_accum[4] = {
    make_float4(0),
    make_float4(0),
    make_float4(0),
    make_float4(0) 
  };

  for(int xbucket=0; xbucket<2; ++xbucket) {
    for(int ybucket=0; ybucket<2; ++ybucket) {
      totbucket = 2*xbucket+ybucket;
      //int spp_hemi = indirect_spp[make_uint2(bucket_index.x, bucket_index.y+totbucket)];
      int spp_hemi = 2;
      uint2 cur_bucket_index = make_uint2(launch_index.x,launch_index.y*4 +totbucket);
      for(int a=0; a<spp_hemi; ++a)
      {
        for(int b=0; b<spp_hemi; ++b)
        {
          seed.x = rot_seed(seed.x, a);
          seed.y = rot_seed(seed.y, b);
          sample.x = (xbucket + (rnd(seed.x)+((float)a))/spp_hemi)/2.;
          sample.y = (ybucket + (rnd(seed.y)+((float)b))/spp_hemi)/2.;

          //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
          sampleUnitHemisphere( sample, u,v,w, sampleDir);

          //construct indirect sample
          PerRayData_indirect indirect_prd;
          indirect_prd.hit = false;
          indirect_prd.color = make_float3(0,0,0);
          indirect_prd.distance = 100000000;


          optix::Ray indirect_ray ( prd.world_loc, sampleDir, indirect_ray_type, 0.001);//scene_epsilon );

          rtTrace(top_object, indirect_ray, indirect_prd);

          if(indirect_prd.hit) {
            float3 zvec = indirect_prd.distance*sampleDir;
            float cur_zpmin = sqrt(dot(zvec,zvec) - dot(prd.n,zvec));
            zpmin[totbucket] = min(zpmin[totbucket], cur_zpmin);
            zpmax[totbucket] = max(zpmax[totbucket], cur_zpmin);
            //z_perp[cur_bucket_index].x = min(z_perp[cur_bucket_index].x, cur_zpmin);
            //z_perp[cur_bucket_index].y = max(z_perp[cur_bucket_index].y, cur_zpmin);
            // TODO: find actual zpmax
            //prd_direct.zpmax = max(prd_direct.zpmax, cur_zpmin);
          }

          //nDl term needed if sampling by cosine density?
          //float nDl = max(dot( ffnormal, normalize(sampleDir)), 0.f);

          float3 H = normalize(sampleDir - ray.direction);
          float3 R = normalize( 2*prd.n*(dot(prd.n,sampleDir)) - sampleDir);
          float nDr = max(dot( normalize(-ray.direction), R), 0.f);
          float nDh = max(dot( prd.n, H ),0.0f);
          float nDl = max(dot( prd.n, sampleDir),0.f);
          float3 cur_ind = make_float3(0);
          if (nDl > 0.01)
          {
              cur_ind = prd.Kd * indirect_prd.color;
              
              if (nDl > 0.01)
              {
                cur_ind += M_PI * prd.Ks *indirect_prd.color* pow(nDr, prd.phong_exp)/nDl;
              }
          }

          //indirectColor += Kd * indirect_prd.color;


          //TODO: optimize
          cur_indirect_accum[totbucket].x += cur_ind.x;
          cur_indirect_accum[totbucket].y += cur_ind.y;
          cur_indirect_accum[totbucket].z += cur_ind.z;

        }
      }
      //cur_indirect_accum[totbucket].w = spp_hemi*spp_hemi;
      //indirect_spp[cur_bucket_index] = spp_hemi*spp_hemi;

    }
  }

  float3 indirect_illum_unavg = make_float3(0);
  float indirect_illum_num = 0;

  for (int i = 0; i < 4; ++i)
  {
    uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+i);
    float3 cur_indirect_illum_unavg = make_float3(
      cur_indirect_accum[i].x,
      cur_indirect_accum[i].y,
      cur_indirect_accum[i].z);
    //indirect_illum_accum[cur_bucket_index] += cur_indirect_accum[i];
    //indirect_illum_sep[cur_bucket_index] = cur_indirect_illum_unavg/cur_indirect_accum[i].w;
    indirect_illum_unavg += indirect_illum_sep[cur_bucket_index];
    //indirect_illum_unavg += cur_indirect_illum_unavg;
    //indirect_illum_num += cur_indirect_accum[i].w;
    
    z_perp[cur_bucket_index].x = min(z_perp[cur_bucket_index].x, zpmin[i]);
    z_perp[cur_bucket_index].y = max(z_perp[cur_bucket_index].y, zpmax[i]);

    //target_indirect_spp[cur_bucket_index] = 0;
    //indirect_spp[cur_bucket_index] = init_ind_spp_sqrt/2;
  }
  //if (indirect_illum_num > 0.01)
    //indirect_illum[launch_index] = indirect_illum_unavg/indirect_illum_num;
    indirect_illum[launch_index] = indirect_illum_unavg/4;
  indirect_rng_seeds[launch_index] = seed;


  // Visualize projected distances
  //direct_illum[launch_index] = make_float3( proj_dist/10.0 );

  // Visualize SPP
  //direct_illum[launch_index] = heatMap(spp/1000.);
  
  // SPP
  //assuming 1:1 aspect
  for (int i = 0; i < 4; ++i) {
    uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+i);

    float zpmin = z_perp[cur_bucket_index].x;
    float zpmax = z_perp[cur_bucket_index].y;

    float proj_dist = 2./image_dim.y * prd.t_hit*tan(fov/2.*M_PI/180.);
    float wvmax = omega_v_max[launch_index];
    float alpha = 0.5;
    float spp_term1 = proj_dist * wvmax/zpmin + alpha;
    float spp_term2 = 1+zpmax/zpmin;

    float spp = spp_term1*spp_term1 * wvmax*wvmax * spp_term2*spp_term2;

    target_indirect_spp[cur_bucket_index] = spp;

  }
}

RT_PROGRAM void pinhole_camera_continued_sample() {
  float3 ray_origin = world_loc[launch_index];
  float3 cur_n = n[launch_index];

  int spp_sqrt = 4;
  //int cur_spp = indirect_spp[launch_index];
  //int target_spp = target_indirect_spp[launch_index];

  //if (cur_spp > target_spp || cur_spp > max_spp)
  //  return;
  //indirect_spp[launch_index] += spp_sqrt*spp_sqrt;

  
  uint2 seed = indirect_rng_seeds[launch_index];
  float3 u,v,w;
  float3 u2,v2,w2;
  float3 sampleDir;
  createONB(cur_n, u,v,w);
  float3 eye_to_loc = normalize(ray_origin-eye);
  float3 perf_refl_r = eye_to_loc - 2*cur_n*dot(cur_n, eye_to_loc);
  createONB(perf_refl_r, u2,v2,w2);
  float2 sample = make_float2(0);
  
  int xbucket = 0;
  int ybucket = 0;
  int totbucket = 0;

  float zpmin[4] = { 100000, 100000, 100000, 100000 };
  float zpmax[4] = { 0, 0, 0, 0 };

  float4 cur_indirect_accum[4] = {
    make_float4(0),
    make_float4(0),
    make_float4(0),
    make_float4(0) 
  };
  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);

  int sample_type = 0; // quick addition: 0 = nDl, 1 = nDr (weighted w/ power)
  float3 cur_Ks = image_Ks[launch_index];
  if (cur_Ks.x + cur_Ks.y + cur_Ks.z > 0.01)
  {
    sample_type = 1;
  }

  for(int i=0; i<2; ++i) {
    seed.x = rot_seed(seed.x, i);
    if(i > 0)
      xbucket = 1;
    ybucket = 0;
    for(int j=0; j<2; ++j) {
      seed.y = rot_seed(seed.y, j);
      if(j > 0)
        ybucket = 1;
      totbucket = 2*xbucket+ybucket;
      //int spp_hemi = indirect_spp[make_uint2(bucket_index.x, bucket_index.y+totbucket)];
      int spp_hemi = 2;
      if (indirect_spp[make_uint2(bucket_index.x, bucket_index.y+totbucket)] >
          min(max_spp, target_indirect_spp[make_uint2(bucket_index.x, bucket_index.y+totbucket)]))
        spp_hemi = 0;
      //spp_hemi = 4;
      //spp_hemi = 2;
      uint2 cur_bucket_index = make_uint2(launch_index.x,launch_index.y*4 +totbucket);
      for(int a=0; a<spp_hemi; ++a)
      {
        for(int b=0; b<spp_hemi; ++b)
        {
          sample.x = (xbucket + (rnd(seed.x)+((float)a))/spp_hemi)/2.;
          sample.y = (ybucket + (rnd(seed.y)+((float)b))/spp_hemi)/2.;
          //sample.x = (rnd(seed.x)+((float)k))/spp_hemi;
          //sample.y = (rnd(seed.y)+((float)k))/spp_hemi;
          //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
          if (sample_type == 1)
            sampleUnitHemispherePower( sample, u2,v2,w2, image_phong_exp[launch_index], sampleDir);
            //sampleUnitHemispherePower( sample, u2,v2,w2, 1, sampleDir);
          else
            sampleUnitHemisphere( sample, u,v,w, sampleDir);

          //construct indirect sample
          PerRayData_indirect indirect_prd;
          indirect_prd.hit = false;
          indirect_prd.color = make_float3(0,0,0);
          indirect_prd.distance = 100000000;


          optix::Ray indirect_ray ( ray_origin, sampleDir, indirect_ray_type, 0.01);//scene_epsilon );

          rtTrace(top_object, indirect_ray, indirect_prd);

          if(indirect_prd.hit) {
            float3 zvec = indirect_prd.distance*sampleDir;
            float cur_zpmin = sqrt(dot(zvec,zvec) - dot(cur_n,zvec));
            zpmin[totbucket] = min(zpmin[totbucket], cur_zpmin);
            zpmax[totbucket] = max(zpmax[totbucket], cur_zpmin);
            //z_perp[cur_bucket_index].x = min(z_perp[cur_bucket_index].x, cur_zpmin);
            //z_perp[cur_bucket_index].y = max(z_perp[cur_bucket_index].y, cur_zpmin);
            // TODO: find actual zpmax
            //prd_direct.zpmax = max(prd_direct.zpmax, cur_zpmin);
          }

          //nDl term needed if sampling by cosine density?
          //float nDl = max(dot( ffnormal, normalize(sampleDir)), 0.f);
          float3 R = normalize( 2*cur_n*dot(cur_n, sampleDir) - sampleDir);
          float nDr = max(dot(normalize(eye-ray_origin), R), 0.f);

          float3 H = normalize(sampleDir + (eye-ray_origin));
          float nDh = max(dot( cur_n, H ),0.0f);
          float nDl = max(dot( cur_n, sampleDir),0.f);
          float3 cur_ind = make_float3(0);
          if (nDl > 0.0)
          {
            float3 diff_term = image_Kd[launch_index] * indirect_prd.color * M_PIf;
            if (sample_type == 1)
              diff_term *= nDl/pow(nDr, image_phong_exp[launch_index]) * 2/(image_phong_exp[launch_index]+1);
            cur_ind += diff_term;
            if (nDr > 0.01)
            {
              float3 spec_term = image_Ks[launch_index] *indirect_prd.color * M_PIf;
              if (sample_type == 0)
                spec_term *= pow(nDr, image_phong_exp[launch_index]);
              else {
                spec_term *= nDl * 2./(image_phong_exp[launch_index]+1);
              }
              cur_ind += spec_term;
            }
          }

          //indirectColor += Kd * indirect_prd.color;


          //TODO: optimize
          cur_indirect_accum[totbucket].x += cur_ind.x;
          cur_indirect_accum[totbucket].y += cur_ind.y;
          cur_indirect_accum[totbucket].z += cur_ind.z;
        }
      }
      cur_indirect_accum[totbucket].w = spp_hemi*spp_hemi;
      indirect_spp[cur_bucket_index] += spp_hemi*spp_hemi;
      
    }
  }
  
  float3 indirect_illum_unavg = make_float3(0);
  float indirect_illum_num = 0;


  for (int i = 0; i < 4; ++i)
  {
    uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+i);
      float3 cur_indirect_illum_unavg = make_float3(
        cur_indirect_accum[i].x,
        cur_indirect_accum[i].y,
        cur_indirect_accum[i].z);
        indirect_illum_accum[cur_bucket_index] += cur_indirect_accum[i];
      float3 cum_indirect_illum_unavg = make_float3(
        indirect_illum_accum[cur_bucket_index].x,
        indirect_illum_accum[cur_bucket_index].y,
        indirect_illum_accum[cur_bucket_index].z);
        indirect_illum_sep[cur_bucket_index] = cum_indirect_illum_unavg/indirect_illum_accum[cur_bucket_index].w;
    z_perp[cur_bucket_index].x = min(z_perp[cur_bucket_index].x, zpmin[i]);
    z_perp[cur_bucket_index].y = max(z_perp[cur_bucket_index].y, zpmax[i]);
    //indirect_illum_unavg += cum_indirect_illum_unavg;
    //indirect_illum_num += indirect_illum_accum[cur_bucket_index].w;
    
    indirect_illum_unavg += indirect_illum_sep[cur_bucket_index];
  }
  
  //if (indirect_illum_num > 0.01)
    //indirect_illum[launch_index] = indirect_illum_unavg/indirect_illum_num;
indirect_illum[launch_index] = indirect_illum_unavg/4;
  indirect_rng_seeds[launch_index] = seed;

  
}



RT_PROGRAM void display_camera() {

  if (direct_illum[launch_index].x < -1.0f) {
    output_buffer[launch_index] = make_color( bg_color );
    return;
  }

  output_buffer[launch_index] = make_color( direct_illum[launch_index]+indirect_illum[launch_index]);
  if (filter_indirect)
    output_buffer[launch_index] = make_color( direct_illum[launch_index]+indirect_illum_filt_int[launch_index]);

  if (view_mode) {
    uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);
    if (view_mode == 1)
      output_buffer[launch_index] = make_color( direct_illum[launch_index] );
    if (view_mode == 2) 
    {
      if (filter_indirect)
        output_buffer[launch_index] = make_color( indirect_illum_filt_int[launch_index]);
      else
        output_buffer[launch_index] = make_color( indirect_illum[launch_index]);
    }
    if (view_mode == 3) 
      output_buffer[launch_index] = make_color( heatMap( omega_v_max[launch_index]/5. ));
    
    if (view_mode == 4)
      output_buffer[launch_index] = make_color( make_float3(use_filter[launch_index]) );
    if (view_mode == 5 || view_mode == 6 || view_mode == 7 || view_mode == 8) {
      int sep_ind = view_mode-5;
      if (filter_indirect)
        output_buffer[launch_index] = make_color( indirect_illum_filt[make_uint2(
              launch_index.x, launch_index.y*4+sep_ind )]);
      else
        output_buffer[launch_index] = make_color( indirect_illum_sep[make_uint2(
              launch_index.x, launch_index.y*4+sep_ind )]);
    }
    if (view_mode == 9 || view_mode == 10 || view_mode == 11 || view_mode == 12) {
      int sep_ind = view_mode-9;
      output_buffer[launch_index] = make_color( heatMap(z_perp[make_uint2(
              bucket_index.x, bucket_index.y+sep_ind)].x/200.));
    }
    if (view_mode == 13)
      output_buffer[launch_index] = make_color( make_float3(
            indirect_spp[launch_index] > target_indirect_spp[launch_index]) );
    if (view_mode == 14 || view_mode == 15 || view_mode == 16 || view_mode == 17) {
      int sep_ind = view_mode-14;
      output_buffer[launch_index] = make_color( heatMap( target_indirect_spp[make_uint2(
              bucket_index.x, bucket_index.y+sep_ind)]/1000. ));
    }
    if (view_mode == 18 || view_mode == 19 || view_mode == 20 || view_mode == 21) {
      int sep_ind = view_mode-18;
      output_buffer[launch_index] = make_color( heatMap( indirect_spp[make_uint2(
              bucket_index.x, bucket_index.y+sep_ind)]/1000. ));
    }
    //filt radius = 1/zpmin
    //spp
  }

  //output_buffer[launch_index] = make_color( image_Ks[launch_index] );
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
  const float angle_threshold = 1.f * M_PI/180.0f;

  if (i > 0 && i < buf_size.x && j > 0 && j < buf_size.y) {
    uint2 target_index = make_uint2(i,j);
    uint2 target_bucket_index = make_uint2(i,4*j+bucket);
    float3 target_indirect = indirect_illum_sep[target_bucket_index];
    if (pass == 1)
      target_indirect = indirect_illum_blur1d[target_bucket_index];
    float target_zpmin = z_perp[target_bucket_index].x;
    float3 target_n = n[target_index];
    //bool use_filt = use_filt_indirect[target_index];
    bool use_filt = use_filter[target_index];

    if (use_filt 
        && acos(dot(target_n, cur_n)) < angle_threshold
        //&& abs((1./target_zpmin - 1./cur_zpmin)/(1./target_zpmin + 1./cur_zpmin)) < z_thres
        //&& abs((target_zpmin - cur_zpmin)/(target_zpmin + cur_zpmin)) < z_thres
        //&& (1/target_zpmin - 1/cur_zpmin) < dist_scale_threshold
        )
    {
      float3 target_loc = world_loc[target_index];
      float3 diff = cur_world_loc - target_loc;
      float euclidean_distsq = dot(diff,diff);

      if (euclidean_distsq < (dist_threshold*dist_threshold))
      {
        float weight = gaussFilter(euclidean_distsq, omega_v_max[launch_index]/cur_zpmin);

        blurred_indirect_sum += weight * target_indirect;
        sum_weight += weight;
      }
    }
    
  }
}

RT_PROGRAM void indirect_filter_first_pass()
{
  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);
  for(int bucket = 0; bucket < 4; ++bucket)
  {
    uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+bucket);
    float3 cur_indirect = indirect_illum_sep[cur_bucket_index];
    float cur_zpmin = z_perp[cur_bucket_index].x;
    size_t2 buf_size = indirect_illum_sep.size();
    float3 blurred_indirect = cur_indirect;

    float3 blurred_indirect_sum = make_float3(0.);
    float sum_weight = 0.f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];

    for (int i = -pixel_radius.x; i < pixel_radius.x; i++) 
    {
      if (use_filter[launch_index]) 
        indirectFilter(blurred_indirect_sum, sum_weight,
            cur_world_loc, cur_n, cur_zpmin, launch_index.x+i, launch_index.y,
            buf_size, 0, bucket);
    }

    if (sum_weight > 0.0001f)
      blurred_indirect = blurred_indirect_sum / sum_weight;
    indirect_illum_blur1d[cur_bucket_index] = blurred_indirect;
  }
}

RT_PROGRAM void indirect_filter_second_pass()
{
  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);
  float3 avg_indirect = make_float3(0);
  float tot_spp_pix = 0;
  for(int bucket = 0; bucket < 4; ++bucket)
  {
    uint2 cur_bucket_index = make_uint2(bucket_index.x, bucket_index.y+bucket);
    float3 cur_indirect = indirect_illum_blur1d[cur_bucket_index];
    float cur_zpmin = z_perp[cur_bucket_index].x;
    size_t2 buf_size = indirect_illum_blur1d.size();
    float3 blurred_indirect = cur_indirect;

    float3 blurred_indirect_sum = make_float3(0.);
    float sum_weight = 0.f;

    float3 cur_world_loc = world_loc[launch_index];
    float3 cur_n = n[launch_index];

    for (int j = -pixel_radius.y; j < pixel_radius.y; j++) 
    {
      if (use_filter[launch_index]) 
        indirectFilter(blurred_indirect_sum, sum_weight,
            cur_world_loc, cur_n, cur_zpmin, launch_index.x, launch_index.y+j,
            buf_size, 1, bucket);
    }

    if (sum_weight > 0.0001f)
      blurred_indirect = blurred_indirect_sum / sum_weight;
    indirect_illum_filt[cur_bucket_index] = blurred_indirect;

    avg_indirect += blurred_indirect;
  }
  avg_indirect /= 4;
  indirect_illum_filt_int[launch_index] = avg_indirect;
  //indirect_illum_filt_int[launch_index] = make_float3(tot_spp_pix)/100.;
}

//
// Returns solid color for miss rays
//
RT_PROGRAM void miss()
{
  prd_direct.color = bg_color;
}

//
// Phong surface shading with shadows
//
rtDeclareVariable(float3,   Ka, , );
rtDeclareVariable(float3,   Ks, , );
rtDeclareVariable(float,    phong_exp, , );
//rtDeclareVariable(float3,   Kd, , );
rtDeclareVariable(int,      obj_id, , );
rtDeclareVariable(float3,   ambient_light_color, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );


RT_PROGRAM void any_hit_shadow()
{
  prd_shadow.hit = true;
  prd_shadow.attenuation = make_float3(0);
  prd_shadow.distance = t_hit;
}

//
// Calculates indirect color
//
RT_PROGRAM void any_hit_indirect()
{
  float2 uv                     = make_float2(texcoord);
  float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));

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

  //shadow
  optix::Ray shadow_ray ( hit_point, L, shadow_ray_type, 0.001);

  PerRayData_shadow shadow_prd;
  shadow_prd.hit = false;
  rtTrace(top_shadower, shadow_ray, shadow_prd);
  if (shadow_prd.hit &&( shadow_prd.distance*shadow_prd.distance) < dot(to_light,to_light)) {
    prd_indirect.color = make_float3(0);
  }

}

//Initial samples
RT_PROGRAM void closest_hit_direct()
{
  float2 uv                     = make_float2(texcoord);

  float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_direct.t_hit = t_hit;
  prd_direct.world_loc = hit_point;
  prd_direct.hit = true;
  prd_direct.n = ffnormal;
  prd_direct.Kd = Kd;
  prd_direct.Ks = Ks;
  prd_direct.phong_exp = phong_exp;

  
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
  float3 to_camera = -ray.direction;
  //temporary - white light
  float3 Lc = make_float3(1,1,1);
  color += Kd * nDl * Lc;// * strength;
  omega_v_max[launch_index] = 2;
  if (nDh > 0 && Ks.x+Ks.y+Ks.z > 0)
  {
    color += Ks * pow(nDh, phong_exp);
    omega_v_max[launch_index] = glossy_blim(H, to_camera, phong_exp);
  }
  prd_direct.color = color;

  //shadow
  optix::Ray shadow_ray ( hit_point, L, shadow_ray_type, 0.01);

  PerRayData_shadow shadow_prd;
  shadow_prd.hit = false;
  rtTrace(top_shadower, shadow_ray, shadow_prd);
  if (shadow_prd.hit &&( shadow_prd.distance*shadow_prd.distance) < dot(to_light,to_light)) {
    prd_direct.color = make_float3(0);
  }
}


//Rays from camera
#if 0
RT_PROGRAM void closest_hit_radiance()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;
  prd_direct.t_hit = t_hit;
  prd_direct.world_loc = hit_point;
  prd_direct.hit = true;
  prd_direct.n = ffnormal;



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
  prd_direct.color = color;

  //shadow
  optix::Ray shadow_ray ( hit_point, L, shadow_ray_type, 0.001);

  PerRayData_shadow shadow_prd;
  shadow_prd.hit = false;
  rtTrace(top_shadower, shadow_ray, shadow_prd);
  if (shadow_prd.hit &&( shadow_prd.distance*shadow_prd.distance) < dot(to_light,to_light)) {
    prd_direct.color = make_float3(0);
  }

  //indirect values
  int sample_sqrt = 2;
  float3 u,v,w;
  float3 sampleDir;
  createONB(ffnormal, u,v,w);
  uint2 bucket_index = make_uint2(launch_index.x, launch_index.y*4);
  //float4 curIndAccum = indirect_illum_accum[launch_index];
  //float3 indirectColor = make_float3(curIndAccum.x, curIndAccum.y, curIndAccum.z);
  float2 sample = make_float2(0);
  int xbucket = 0;
  int ybucket = 0;
  int totbucket = 0;
  //stratify
  for(int i=0; i<sample_sqrt; ++i) {
    seed.x = rot_seed(seed.x, i);
    sample.x = (rnd(seed.x)+((float)i))/sample_sqrt;
    if(i > sample_sqrt/2-1)
      xbucket = 1;
    ybucket = 0;
    for(int j=0; j<sample_sqrt; ++j) {
      seed.y = rot_seed(seed.y, j);
      sample.y = (rnd(seed.y)+((float)j))/sample_sqrt;
      if(j > sample_sqrt/2-1)
        ybucket = 1;
      totbucket = 2*xbucket+ybucket;
      //float2 sample = make_float2( rnd(seed.x), rnd(seed.y) );
      sampleUnitHemisphere( sample, u,v,w, sampleDir);

      //construct indirect sample
      PerRayData_indirect indirect_prd;
      indirect_prd.hit = false;
      indirect_prd.color = make_float3(0,0,0);
      indirect_prd.distance = 100000000;


      optix::Ray indirect_ray ( hit_point, sampleDir, indirect_ray_type, 0.001);//scene_epsilon );

      rtTrace(top_shadower, indirect_ray, indirect_prd);
      uint2 cur_bucket_index = make_uint2(launch_index.x,launch_index.y*4 +totbucket);

      if(indirect_prd.hit) {
        float3 zvec = indirect_prd.distance*sampleDir;
        float cur_zpmin = sqrt(dot(zvec,zvec) - dot(ffnormal,zvec));
        z_perp[cur_bucket_index].x = min(z_perp[cur_bucket_index].x, cur_zpmin);
        z_perp[cur_bucket_index].y = max(z_perp[cur_bucket_index].y, cur_zpmin);
        // TODO: find actual zpmax
        //prd_direct.zpmax = max(prd_direct.zpmax, cur_zpmin);
      }

      //nDl term needed if sampling by cosine density?
      //float nDl = max(dot( ffnormal, normalize(sampleDir)), 0.f);
      float3 cur_ind = Kd * indirect_prd.color;


      //TODO: optimize
      indirect_illum_accum[cur_bucket_index].x += cur_ind.x;
      indirect_illum_accum[cur_bucket_index].y += cur_ind.y;
      indirect_illum_accum[cur_bucket_index].z += cur_ind.z;
      indirect_illum_accum[cur_bucket_index].w += 1;
      
    }
  }

  float num = indirect_illum_accum[launch_index].w \
              + (sample_sqrt*sample_sqrt);
  //prd_direct.indirect_spp += sample_sqrt * sample_sqrt;


  /*
  indirect_illum_accum[launch_index].w = num;
  indirect_illum_accum[launch_index].x = indirectColor.x;
  indirect_illum_accum[launch_index].y = indirectColor.y;
  indirect_illum_accum[launch_index].z = indirectColor.z;
  */

  //avg diff hemispheres
  float3 indirect_color = make_float3(0);
  float num_samp = 0;
  for(int i = 0; i < 4; ++i) {
    uint2 cur_bucket_index = make_uint2(
        bucket_index.x,
        bucket_index.y+i);
    float3 cur_indirect_color = make_float3(
        indirect_illum_accum[cur_bucket_index].x,
        indirect_illum_accum[cur_bucket_index].y,
        indirect_illum_accum[cur_bucket_index].z);
    float cur_num_samp = indirect_illum_accum[cur_bucket_index].w;

    indirect_color += cur_indirect_color;
    num_samp += cur_num_samp;

    indirect_illum_sep[cur_bucket_index] = cur_indirect_color / cur_num_samp;
  }

  //prd_direct.indirect = indirect_color/num_samp;

  //prd_direct.indirect = indirect_illum_accum[bucket_index].xindirect_illum_accum[bucket_index];

  indirect_rng_seeds[launch_index] = seed;

}
#endif

//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
