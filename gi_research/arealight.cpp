/*
* arealight.cpp
* Area Light Filtering
* Adapted from NVIDIA OptiX Tutorial
* Brandon Wang, Soham Mehta
*/

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <ImageLoader.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <math.h>
#include <time.h>
#include <limits>
#include "random.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Mouse.h>

#define SCENE 1
//Grids 1
//Balance 2
//Tentacles 3
//Spheres 4

using namespace optix;

class Arealight : public SampleScene
{
public:
  Arealight(const std::string& texture_path)
    : SampleScene(), _width(1080u), _height(720u), texture_path( texture_path )
    , _frame_number( 0 ), _keep_trying( 1 )
  {
    // reserve some space for timings vector
    _timings.reserve(4);
    _benchmark_iter = 0;
    _benchmark_timings.reserve(4);
  }

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  void   setDimensions( const unsigned int w, const unsigned int h ) { _width = w; _height = h; }
  Buffer getOutputBuffer();

  virtual bool   keyPressed(unsigned char key, int x, int y);

private:
  std::string texpath( const std::string& base );
  void resetAccumulation();
  void createGeometry();

  bool _accel_cache_loaded;

  unsigned int  _frame_number;
  unsigned int  _keep_trying;

  Buffer       _brdf;
  Buffer       _vis;
  GeometryGroup geomgroup;
  GeometryGroup geomgroup2;

  Buffer _conv_buffer;

  unsigned int _width;
  unsigned int _height;
  std::string   texture_path;
  std::string  _ptx_path;

  float _env_theta;
  float _env_phi;

  uint _blur_occ;
  uint _blur_wxf;
  uint _err_vis;
  uint _view_mode;
  uint _lin_sep_blur;

  uint _normal_rpp;
  uint _brute_rpp;
  uint _max_rpp_pass;
  uint _show_progressive;
  int2 _pixel_radius;
  int2 _pixel_radius_wxf;

  float _zmin_rpp_scale;
  bool _converged;

  double _perf_freq;
  std::vector<double> _timings;

  Buffer testBuf;

  AreaLight * _env_lights;
  uint _show_brdf;
  uint _show_occ;
  float _sigma;

  Buffer light_buffer;

  int _benchmark_iter;
  std::vector<double> _benchmark_timings;

  float _anim_t;
  double _previous_frame_time;
  bool _is_anim;

  Transform _trans;
  Transform _trans2;
  Geometry _anim_geom;
  GeometryGroup _anim_geom_group;
  Group _top_grp;

  float _total_avg_cur_spp;
};

Arealight* _scene;
int output_num = 0;

void Arealight::initScene( InitialCameraData& camera_data )
{
  _anim_t = 0;
  _total_avg_cur_spp = 0;
  sutilCurrentTime(&_previous_frame_time);
  _is_anim = true;
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "arealight.cu";
  _ptx_path = ptxpath( "arealight", ss.str() );

  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 7 );
  m_context->setStackSize( 8000 );

  m_context["max_depth"]->setInt(100);
  m_context["radiance_ray_type"]->setUint(0);
  m_context["shadow_ray_type"]->setUint(1);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["importance_cutoff"]->setFloat( 0.01f );
  m_context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, _width, _height) );

  //[bmw] my stuff
  //seed rng
  //(i have no idea if this is right)
  //fix size later
  Buffer shadow_rng_seeds = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, _width, _height);
  m_context["shadow_rng_seeds"]->set(shadow_rng_seeds);
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  // BRDF buffer
#ifdef SPP_STATS
  _brdf = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, _width, _height );
#else
  _brdf = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
#endif
  m_context["brdf"]->set( _brdf );

  // Occlusion buffer
  _vis = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["vis"]->set( _vis );

  // Blurred (on one dimension) cclusion accumulation buffer
  Buffer _occ_blur = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["vis_blur1d"]->set( _occ_blur );

  // samples per pixel buffer
#ifdef SPP_STATS
  Buffer spp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, _width, _height );
#else
  Buffer spp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
#endif
  m_context["spp"]->set( spp );

  // current samples per pixel buffer
#ifdef SPP_STATS
  Buffer spp_cur = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, _width, _height );
#else
  Buffer spp_cur = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
#endif
  m_context["spp_cur"]->set( spp_cur );

  Buffer slope = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height );
  m_context["slope"]->set( slope );

  // gauss values
  Buffer gauss_lookup = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 65);
  m_context["gaussian_lookup"]->set( gauss_lookup );

  float* lookups = reinterpret_cast<float*>( gauss_lookup->map() );
  const float gaussian_lookup[65] = { 0.85, 0.82, 0.79, 0.76, 0.72, 0.70, 0.68,
    0.66, 0.63, 0.61, 0.59, 0.56, 0.54, 0.52,
    0.505, 0.485, 0.46, 0.445, 0.43, 0.415, 0.395,
    0.38, 0.365, 0.35, 0.335, 0.32, 0.305, 0.295,
    0.28, 0.27, 0.255, 0.24, 0.23, 0.22, 0.21,
    0.2, 0.19, 0.175, 0.165, 0.16, 0.15, 0.14,
    0.135, 0.125, 0.12, 0.11, 0.1, 0.095, 0.09,
    0.08, 0.075, 0.07, 0.06, 0.055, 0.05, 0.045,
    0.04, 0.035, 0.03, 0.02, 0.018, 0.013, 0.008,
    0.003, 0.0 };
  const float exp_lookup[60] = {1.0000,    0.9048,    0.8187,    0.7408,    
    0.6703,    0.6065,    0.5488,    0.4966,    0.4493,    0.4066,   
    0.3679,    0.3329,    0.3012,    0.2725,    0.2466,    0.2231,   
    0.2019,    0.1827,    0.1653,    0.1496,    0.1353,    0.1225,    
    0.1108,    0.1003,    0.0907,    0.0821,    0.0743,   0.0672,    
    0.0608,    0.0550,    0.0498,    0.0450,    0.0408,    0.0369,    
    0.0334,    0.0302,    0.0273,    0.0247,    0.0224,    0.0202,   
    0.0183,    0.0166,    0.0150,    0.0136,    0.0123,    0.0111,    
    0.0101,    0.0091,    0.0082,    0.0074,    0.0067,    0.0061,    
    0.0055,    0.0050,    0.0045,    0.0041,    0.0037,    0.0033,    
    0.0030,    0.0027 };

  for(int i=0; i<65; i++) {
    lookups[i] = gaussian_lookup[i];
  }/*
   for(int i=0; i<60; i++) {
   lookups[i] = exp_lookup[i];
   }*/
  gauss_lookup->unmap();

  // world space buffer
  Buffer world_loc = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["world_loc"]->set( world_loc );

  Buffer n = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["n"]->set( n );

  Buffer filter_n = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_n"]->set( filter_n );

  Buffer filter_occ = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_occ"]->set( filter_occ );

  Buffer filter_occ_filter1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["use_filter_occ_filter1d"]->set( filter_occ_filter1d );

  Buffer s1s2_blur1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height );
  m_context["slope_filter1d"]->set( s1s2_blur1d );

  Buffer spp_blur1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["spp_filter1d"]->set( spp_blur1d );

  Buffer dist_to_light = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["dist_to_light"]->set( dist_to_light );

  Buffer proj_d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["proj_d"]->set( proj_d );

  Buffer obj_id = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_INT, _width, _height );
  m_context["obj_id_b"]->set( obj_id );

  _blur_occ = 1;
  m_context["blur_occ"]->setUint(_blur_occ);

  _blur_wxf = 0;
  m_context["blur_wxf"]->setUint(_blur_wxf);

  _err_vis = 1;
  m_context["err_vis"]->setUint(_err_vis);

  _view_mode = 0;
  m_context["view_mode"]->setUint(_view_mode);

  _show_brdf = 1;
  m_context["show_brdf"]->setUint(_show_brdf);

  _show_occ = 1;
  m_context["show_occ"]->setUint(_show_occ);


  _normal_rpp = 3;
  _brute_rpp = 2000;
  _max_rpp_pass = 8;
  float spp_mu = 2;

  m_context["normal_rpp"]->setUint(_normal_rpp);
  m_context["brute_rpp"]->setUint(_brute_rpp);
  m_context["max_rpp_pass"]->setUint(_max_rpp_pass);

  m_context["spp_mu"]->setFloat(spp_mu);

  _zmin_rpp_scale = 1;
  m_context["zmin_rpp_scale"]->setFloat(_zmin_rpp_scale);

  _pixel_radius = make_int2(10,10);
  m_context["pixel_radius"]->setInt(_pixel_radius);

  _pixel_radius_wxf = make_int2(5,5);
  m_context["pixel_radius_wxf"]->setInt(_pixel_radius_wxf);

  // Sampling program
  std::string camera_name;
  camera_name = "pinhole_camera_initial_sample";

  Program ray_gen_program = m_context->createProgramFromPTXFile( _ptx_path, camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // continual Sampling
  std::string continue_sampling = "pinhole_camera_continue_sample";

  Program continue_sampling_program = m_context->createProgramFromPTXFile( _ptx_path, continue_sampling );
  m_context->setRayGenerationProgram( 6, continue_sampling_program );

  // Occlusion Filter programs
  std::string first_pass_occ_filter_name = "occlusion_filter_first_pass";
  Program first_occ_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    first_pass_occ_filter_name );
  m_context->setRayGenerationProgram( 2, first_occ_filter_program );
  std::string second_pass_occ_filter_name = "occlusion_filter_second_pass";
  Program second_occ_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    second_pass_occ_filter_name );
  m_context->setRayGenerationProgram( 3, second_occ_filter_program );

  // S1, S2 Filter programs
  std::string first_pass_s1s2_filter_name = "s1s2_filter_first_pass";
  Program first_s1s2_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    first_pass_s1s2_filter_name );
  m_context->setRayGenerationProgram( 4, first_s1s2_filter_program );
  std::string second_pass_s1s2_filter_name = "s1s2_filter_second_pass";
  Program second_s1s2_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    second_pass_s1s2_filter_name );
  m_context->setRayGenerationProgram( 5, second_s1s2_filter_program );


  // Display program
  std::string display_name;
  display_name = "display_camera";

  Program display_program = m_context->createProgramFromPTXFile( _ptx_path, display_name );
  m_context->setRayGenerationProgram( 1, display_program );

  // Exception / miss programs
  Program exception_program = m_context->createProgramFromPTXFile( _ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

  //Blur program (i hope)
  //Program blur_program = m_context->createProgramFromPTXFile( _ptx_path, "gaussianBlur" );


  std::string miss_name;
  miss_name = "miss";
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( _ptx_path, miss_name ) );
  const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
  m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

#if SCENE==4
  // spheres

  float3 pos = make_float3(4.5, 8, 7.5);
  float3 pos1 = make_float3(7.5, 8, 7.5);
  float3 pos2 = make_float3(4.5, 11, 12.5);
  /*
  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(3.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 17, 7);
  */
  float3 axis1 = pos1-pos;
  float3 axis2 = pos2-pos;

  float3 norm = cross(axis1,axis2);

  AreaLight lights[] = {
    { pos,
    pos1,
    pos2,
    make_float3(1.0f, 1.0f, 1.0f)
    }
  };

  float3 normed_norm = normalize(norm);
  m_context["lightnorm"]->setFloat(normed_norm);

  _sigma = sqrt(length(norm)/4.0f);
  std::cout << "Sigma: " << _sigma << std::endl;
  //_sigma = 0.35;

  m_context["light_sigma"]->setFloat(_sigma);

  /*
  AreaLight lights[] = {
  { make_float3( 0.0, 6.0, -7.0 ),
  make_float3( 4.0, 6.0, -7.0 ),
  make_float3( 0.0, 8.82842712474619f, -4.171572875 ),
  make_float3(1.0f, 1.0f, 1.0f)
  }
  };*/
  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  //light_buffer->setSize( sizeof(_env_lights)/sizeof(_env_lights[0]) );
  //memcpy(light_buffer->map(), _env_lights, sizeof(_env_lights));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera
  camera_data = InitialCameraData( make_float3( 1.0, 9.78, 5.88 ), // eye
    //camera_data = InitialCameraData( make_float3( -5.1f, 2.1f, -3.1f ), // eye
    make_float3( 1.35, 6.2,  2.0f ), // lookat
    //make_float3( -4.0f, 0.0f,  -2.0f ), // looka
    make_float3( 0.0f, 1.0f,  0.0f ), // up
    60 );                             // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
#endif

#if SCENE==3
  // tentacles2

  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(-3.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 17, 7);
  /*
  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(3.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 17, 7);
  */
  float3 axis1 = pos1-pos;
  float3 axis2 = pos2-pos;

  float3 norm = cross(axis1,axis2);

  AreaLight lights[] = {
    { pos,
    pos1,
    pos2,
    make_float3(1.0f, 1.0f, 1.0f)
    }
  };

  float3 normed_norm = normalize(norm);
  m_context["lightnorm"]->setFloat(normed_norm);

  _sigma = sqrt(length(norm)/4.0f);
  std::cout << "Sigma: " << _sigma << std::endl;
  //_sigma = 0.35;

  m_context["light_sigma"]->setFloat(_sigma);

  /*
  AreaLight lights[] = {
  { make_float3( 0.0, 6.0, -7.0 ),
  make_float3( 4.0, 6.0, -7.0 ),
  make_float3( 0.0, 8.82842712474619f, -4.171572875 ),
  make_float3(1.0f, 1.0f, 1.0f)
  }
  };*/
  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  //light_buffer->setSize( sizeof(_env_lights)/sizeof(_env_lights[0]) );
  //memcpy(light_buffer->map(), _env_lights, sizeof(_env_lights));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera
  camera_data = InitialCameraData( make_float3( -2.1, 8.0, -10 ), // eye
    //camera_data = InitialCameraData( make_float3( -5.1f, 2.1f, -3.1f ), // eye
    make_float3( -20.0f, 4.0f,  -13.0f ), // lookat
    //make_float3( -4.0f, 0.0f,  -2.0f ), // looka
    make_float3( 0.0f, 1.0f,  0.0f ), // up
    60 );                             // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
#endif
#if SCENE==1
  // grids2

  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(1.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 21.8284, 3.8284);
  /*
  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(3.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 17, 7);
  */
  float3 axis1 = pos1-pos;
  float3 axis2 = pos2-pos;

  float3 norm = cross(axis1,axis2);

  AreaLight lights[] = {
    { pos,
    pos1,
    pos2,
    make_float3(1.0f, 1.0f, 1.0f)
    }
  };

  float3 normed_norm = normalize(norm);
  m_context["lightnorm"]->setFloat(normed_norm);

  _sigma = sqrt(length(norm)/4.0f);
  std::cout << "Sigma: " << _sigma << std::endl;
  //_sigma = 0.75;

  m_context["light_sigma"]->setFloat(_sigma);

  /*
  AreaLight lights[] = {
  { make_float3( 0.0, 6.0, -7.0 ),
  make_float3( 4.0, 6.0, -7.0 ),
  make_float3( 0.0, 8.82842712474619f, -4.171572875 ),
  make_float3(1.0f, 1.0f, 1.0f)
  }
  };*/
  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  //light_buffer->setSize( sizeof(_env_lights)/sizeof(_env_lights[0]) );
  //memcpy(light_buffer->map(), _env_lights, sizeof(_env_lights));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera
  camera_data = InitialCameraData( make_float3( -4.5f, 2.5f, 5.5f ), // eye
    //camera_data = InitialCameraData( make_float3( -5.1f, 2.1f, -3.1f ), // eye
    make_float3( 0.0f, 0.5f,  0.0f ), // lookat
    //make_float3( -4.0f, 0.0f,  -2.0f ), // looka
    make_float3( 0.0f, 1.0f,  0.0f ), // up
    60 );                             // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

#endif
#if SCENE==2
  // balance
  // Area lights

  float3 pos = make_float3( 18.5556f, 25.1727f, 10.9409f);
  float3 pos1 = make_float3( 18.5556f, 25.1727f, 13.9409f);
  float3 pos2 = make_float3( 15.6368f, 27.5674f, 10.9431f);

  /*
  float3 pos = make_float3(-4.5, 16, 8);
  float3 pos1 = make_float3(3.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 17, 7);
  */
  float3 axis1 = pos1-pos;
  float3 axis2 = pos2-pos;

  float3 norm = cross(axis1,axis2);

  AreaLight lights[] = {
    { pos,
    pos1,
    pos2,
    make_float3(1.0f, 1.0f, 1.0f)
    }
  };

  float3 normed_norm = normalize(norm);
  m_context["lightnorm"]->setFloat(normed_norm);

  _sigma = sqrt(length(norm)/4.0f);
  std::cout << "Sigma: " << _sigma << std::endl;

  m_context["light_sigma"]->setFloat(_sigma);

  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(AreaLight));
  //light_buffer->setSize( sizeof(_env_lights)/sizeof(_env_lights[0]) );
  //memcpy(light_buffer->map(), _env_lights, sizeof(_env_lights));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera
  camera_data = InitialCameraData( make_float3( 10.0f, 12.0f, -9.0f ), // eye
    make_float3( 0.0f, 6.0f,  -7.0f ), // lookat
    make_float3( 0.0f, 1.0f,  0.0f ), // up
    60 );                             // vfov
  /*
  camera_data = InitialCameraData( make_float3( 7.0f, 9.2f, 6.0f ), // eye
  make_float3( 0.0f, 2.0f,  0.0f ), // lookat
  make_float3( 0.0f, 1.0f,  0.0f ), // up
  60.0f );                          // vfov
  */
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
#endif

  _env_theta = 0.0f;
  _env_phi = 0.0f;
  m_context["env_theta"]->setFloat(_env_theta);
  m_context["env_phi"]->setFloat(_env_phi);

  // Populate scene hierarchy
  createGeometry();

  //Initialize progressive accumulation
  resetAccumulation();

  // Prepare to run
  m_context->validate();
  m_context->compile();
}


Buffer Arealight::getOutputBuffer()
{

  return m_context["output_buffer"]->getBuffer();
}

void Arealight::trace( const RayGenCameraData& camera_data )
{
  _frame_number ++;
  /*
  if (_frame_number == 3) {
  std::cout << "Matrix of spp" << std:: endl;
  Buffer spp = m_context["spp"]->getBuffer();
  float* spp_arr = reinterpret_cast<float*>( spp->map() );
  for(unsigned int j = 0; j < _height; ++j ) {
  for(unsigned int i = 0; i < _width; ++i ) {
  std::cout << spp_arr[i+j*_width] <<", ";
  }
  std::cout << std::endl;
  }
  spp->unmap();
  }
  */

  if(m_camera_changed) {
    m_context["numAvg"]->setUint(1);
    m_camera_changed = false;
    resetAccumulation();
    _benchmark_iter = 0;
  }


  double t;
  if(GLUTDisplay::getContinuousMode() != GLUTDisplay::CDNone) {
    sutilCurrentTime(&t);
  } else {
    t = _previous_frame_time;
  }
  
  double time_elapsed = t - _previous_frame_time;

  _previous_frame_time = t;

  if (_is_anim)
    //_anim_t += 0.03; //0.7 * time_elapsed; //0.6 * time_elapsed;
    _anim_t += 0.7 * time_elapsed; //0.6 * time_elapsed;
  float3 eye, u, v, w;
  eye.x = (float) (camera_data.eye.x * sin(_anim_t));
  eye.y = (float)( 0.2 + camera_data.eye.y + cos( _anim_t*1.5 ) );
  eye.z = (float)( 0.5+camera_data.eye.z*cos( _anim_t ) );
  //eye = camera_data.eye;
  float3 lookat = make_float3(0);
//#define CHANGE_LIGHT_SIZE
#ifdef CHANGE_LIGHT_SIZE
  m_context["light_sigma"]->setFloat(2.2+1*sin(_anim_t));
#endif

//#define MOVE_LIGHT
#ifdef MOVE_LIGHT
float3 d = make_float3(0.15*sin(_anim_t/1.0),0.2*cos(_anim_t/1.0),0.1*sin(_anim_t/1.2));
AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
lights[0].v1 -= d;
lights[0].v2 -= d;
lights[0].v3 -= d;

light_buffer->unmap();

//m_camera_changed = true;
#endif

  
#if SCENE==1
  optix::Matrix4x4 tr = optix::Matrix4x4::rotate(sin(_anim_t/6.0)*M_PI,make_float3(0,1,0))
    *optix::Matrix4x4::translate(make_float3(0,0.2+0.3*sin(_anim_t),0));

  optix::Matrix4x4 tr2 = optix::Matrix4x4::rotate(-M_PI/6+cos(_anim_t*2)/2, make_float3(1,0,0));
//#define MOVE_GEOM
#ifdef MOVE_GEOM
  _trans->setMatrix( false, tr.getData(), 0);
  _trans2->setMatrix( false, tr2.getData(), 0);
  _top_grp->getAcceleration()->markDirty();
#endif
  
#endif

#if SCENE==2
  lookat = make_float3( -5.0f, 2.0f,  -12.0f ); // lookat
  eye.x = (float) (camera_data.eye.x * sin(_anim_t/2));
  eye.y = (float)( 8.5 - 5*cos( _anim_t/2 ) );
  eye.z = (float)( 0.5+camera_data.eye.z*cos( _anim_t/2 ) );
#endif
#if SCENE==3
  lookat = make_float3( -20.0f, 4.0f,  -13.0f ); // lookat
  eye.x = (float) (-20.0f + 20* sin(_anim_t));
  eye.y = (float)( 4.0f + 5*sin( _anim_t*1.5 ) );
  eye.z = (float)( -13.0f + 20*cos( _anim_t ) );
#define MOVE_GEOM
#ifdef MOVE_GEOM Matrix4x4 tr = Matrix4x4::translate(make_float3(0,-5,0))
    * Matrix4x4::translate(make_float3(-10,0,-2))
    * Matrix4x4::rotate(-10*M_PI/180, make_float3(0,0,1))
    * Matrix4x4::rotate(35*M_PI/180, make_float3(0,1,0))
    * Matrix4x4::scale(make_float3(2))
    * Matrix4x4::translate(make_float3(-1.558, 0, -3.667))
    * Matrix4x4::rotate(_anim_t,make_float3(0,1,0))
    * Matrix4x4::translate(make_float3(1.558, 0, 3.667));
  _trans->setMatrix( false, tr.getData(), 0);
  _top_grp->getAcceleration()->markDirty();
#endif
#endif

  PinholeCamera pc( eye, lookat, make_float3(0,1,0), 60.f, 60.f/(640.0/480.0) );
  pc.getEyeUVW( eye, u, v, w );
  m_context["eye"]->setFloat( eye );
  m_context["U"]->setFloat( u );
  m_context["V"]->setFloat( v );
  m_context["W"]->setFloat( w );
  //m_camera_changed = true;


#define MOVE_CAMERA
#ifndef MOVE_CAMERA
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
#endif

  // do i need to reseed?
  Buffer shadow_rng_seeds = m_context["shadow_rng_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( shadow_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  shadow_rng_seeds->unmap();

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  m_context["frame"]->setUint( _frame_number );

  int num_resample = ceil((float)_brute_rpp * _brute_rpp / (_max_rpp_pass * _max_rpp_pass));
  //std::cout << "Number of passes to resample: " << num_resample << std::endl;

  //Initial 16 Samples
  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Filter s1,s2
  m_context->launch( 4, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  m_context->launch( 5, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Resample
#if 0
  num_resample = 20;
  for(int i = 0; i < num_resample; i++)
#endif
    m_context->launch( 6, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Filter occlusion
  m_context->launch( 2, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  m_context->launch( 3, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  //Display
 
  /*
  if (_view_mode) {
    if (_view_mode == 2) {
      //scale
      Buffer slope = m_context["slope"]->getBuffer();
      float min_s2 = 100000000.0;
      float max_s2 = 0.0;
      float2* slope_arr = reinterpret_cast<float2*>( slope->map() );
      for(unsigned int j = 0; j < _height; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          //std::cout << spp_arr[i+j*_width] <<", ";
          float cur_s2_val = slope_arr[i+j*_width].y;
          if (cur_s2_val < 9999.0 && cur_s2_val > 0.01) {
            min_s2 = min(min_s2,cur_s2_val);
            max_s2 = max(max_s2,cur_s2_val);
          }
        }
        //std::cout << std::endl;
      }
      m_context["max_disp_val"]->setFloat(max_s2);
      m_context["min_disp_val"]->setFloat(min_s2);
      std::cout << "max,min s2: " << max_s2 << ", " << min_s2 << std::endl;
      slope->unmap();
    }
  }
  */
  m_context->launch( 1, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );

  #define NUM_FRAMES 500
#define SPP_AVG
#ifdef SPP_AVG
  //spp = m_context["spp"]->getBuffer();
  Buffer cur_spp = m_context["spp_cur"]->getBuffer();
  Buffer brdf = m_context["brdf"]->getBuffer();
  float min_cur_spp = 10000000.0;
  float max_cur_spp = 0.0;
  float avg_cur_spp = 0.0;
  float3* brdf_arr = reinterpret_cast<float3*>( brdf->map() );
  uint2 err_loc;
  uint2 err_first_loc;
  bool first_loc_set = false;
  int num_cur_avg = 0;
  int num_cur_low = 0;
  float* cur_spp_arr = reinterpret_cast<float*>( cur_spp->map() );
  for(unsigned int j = 0; j < _height; ++j ) {
    for(unsigned int i = 0; i < _width; ++i ) {
      float cur_brdf_x = brdf_arr[i+j*_width].x;
      if (cur_brdf_x > -1) {
        //std::cout << spp_arr[i+j*_width] <<", ";
        float cur_spp_val = cur_spp_arr[i+j*_width];
        if (cur_spp_val > -0.001) {
          min_cur_spp = min(min_cur_spp,cur_spp_val);
          max_cur_spp = max(max_cur_spp,cur_spp_val);
          avg_cur_spp += cur_spp_val;
          num_cur_avg++;
          if (cur_spp_val < 10)
            num_cur_low++;
        }
      }
    }
    //std::cout << std::endl;
  }
  cur_spp->unmap();
  brdf->unmap();
  avg_cur_spp /= num_cur_avg;
  _total_avg_cur_spp += avg_cur_spp;
  //std::cout << "Average SPP this frame: " << avg_cur_spp << std::endl;

  if (_frame_number > NUM_FRAMES) {
    std::cout << "SPP Average: " << (_total_avg_cur_spp/NUM_FRAMES) << std::endl;
  }
#endif
//#define CAPTURE_FRAMES
#ifdef CAPTURE_FRAMES
  std::stringstream fname;
  fname << "output_";
  fname << std::setw(7) << std::setfill('0') << output_num;
  fname << ".ppm";
  Buffer output_buf = _scene->getOutputBuffer();
  output_num++;

  sutilDisplayFilePPM(fname.str().c_str(), output_buf->get());

  if (output_num%(NUM_FRAMES/10) == 0)
    std::cout << ((float)(output_num)/NUM_FRAMES)*100.0 << "%" << std::endl;  if (output_num > NUM_FRAMES) {
    exit(0);
  }

#endif

  if (_frame_number > NUM_FRAMES) {
    //QueryPerformanceCounter(&ended_render);
    //double timing = (double(ended_render.QuadPart - _started_render.QuadPart)/_perf_freq/1000.0);
    //std::cout << "Finished rendering at " << (NUM_FRAMES/timing) << " fps (average)" << std::endl;
    exit(0);
  }
}


void Arealight::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}

std::string Arealight::texpath( const std::string& base )
{
  return texture_path + "/" + base;
}

float4 make_plane( float3 n, float3 p )
{
  n = normalize(n);
  float d = -dot(n, p);
  return make_float4( n, d );
}

void Arealight::resetAccumulation()
{
  _frame_number = 0;
  m_context["frame"]->setUint( _frame_number );
  _converged = false;
  //_started_render = timeGetTime();
  //QueryPerformanceCounter(&_started_render);

  //_perf_freq = double(freq.QuadPart)/1000.0;
}


bool Arealight::keyPressed(unsigned char key, int x, int y) {
  float delta = 0.5f;

  Buffer spp;
  switch(key) {
  case 'U':
  case 'u':
    {
      float3 d = make_float3(delta,0,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'J':
  case 'j':
    {
      float3 d = make_float3(delta,0,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'I':
  case 'i':
    {
      float3 d = make_float3(0,delta,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'K':
  case 'k':
    {
      float3 d = make_float3(0,delta,0);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;

      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }

  case 'O':
  case 'o':
    {
      float3 d = make_float3(0,0,delta);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 += d;
      lights[0].v2 += d;
      lights[0].v3 += d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'L':
  case 'l':
    {
      float3 d = make_float3(0,0,delta);
      AreaLight* lights = reinterpret_cast<AreaLight*>(light_buffer->map());
      lights[0].v1 -= d;
      lights[0].v2 -= d;
      lights[0].v3 -= d;

      std::cout << "Light now at: " << "\n"
        "v1: " << lights[0].v1.x << "," << lights[0].v1.y << "," << lights[0].v1.z << "\n"
        "v2: " << lights[0].v2.x << "," << lights[0].v2.y << "," << lights[0].v2.z << "\n"
        "v3: " << lights[0].v3.x << "," << lights[0].v3.y << "," << lights[0].v3.z
        << std::endl;

      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }

  case 'M':
  case 'm':
    _show_brdf = 1-_show_brdf;
    m_context["show_brdf"]->setUint(_show_brdf);
    if (_show_brdf)
      std::cout << "BRDF: On" << std::endl;
    else
      std::cout << "BRDF: Off" << std::endl;
    return true;
  case 'N':
  case 'n':
    _show_occ = 1-_show_occ;
    m_context["show_occ"]->setUint(_show_occ);
    if (_show_occ)
      std::cout << "Occlusion Display: On" << std::endl;
    else
      std::cout << "Occlusion Display: Off" << std::endl;
    return true;



  case 'V':
  case 'v':
    {
#ifdef SPP_STATS
      std::cout << "SPP stats" << std:: endl;
      //spp = m_context["spp"]->getBuffer();
      Buffer cur_spp = m_context["spp_cur"]->getBuffer();
      Buffer brdf = m_context["brdf"]->getBuffer();
      spp = m_context["spp"]->getBuffer();
      float min_cur_spp = 10000000.0;
      float max_cur_spp = 0.0;
      float avg_cur_spp = 0.0;
      float min_spp = 100000000.0; //For some reason, numeric_limits gives me something weird in the end? //std::numeric_limits<float>::max();
      float max_spp = 0.0;
      float avg_spp = 0.0;
      //float* spp_arr = reinterpret_cast<float*>( spp->map() );
      float* spp_arr = reinterpret_cast<float*>( spp->map() );
      float3* brdf_arr = reinterpret_cast<float3*>( brdf->map() );
      int num_avg = 0;

      int num_low = 0;
      for(unsigned int j = 0; j < _height; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          //std::cout << spp_arr[i+j*_width] <<", ";
          float cur_brdf_x = brdf_arr[i+j*_width].x;
          if (cur_brdf_x > -1) {
            //std::cout << "brdf: " << cur_brdf_x << std::endl;
            float cur_spp_val = spp_arr[i+j*_width];
            if (cur_spp_val > -0.001) {
              min_spp = min(min_spp,cur_spp_val);
              max_spp = max(max_spp,cur_spp_val);
              avg_spp += cur_spp_val;
              num_avg++;
              if (cur_spp_val < 10)
                num_low++;

            }
          } 
        }
        //std::cout << std::endl;
      }
      spp->unmap();
      avg_spp /= num_avg;
      uint2 err_loc;
      uint2 err_first_loc;
      bool first_loc_set = false;
      int num_cur_avg = 0;
      int num_cur_low = 0;
      float* cur_spp_arr = reinterpret_cast<float*>( cur_spp->map() );
      for(unsigned int j = 0; j < _height; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          float cur_brdf_x = brdf_arr[i+j*_width].x;
          if (cur_brdf_x > -1) {
            //std::cout << spp_arr[i+j*_width] <<", ";
            float cur_spp_val = cur_spp_arr[i+j*_width];
            if (cur_spp_val > -0.001) {
              min_cur_spp = min(min_cur_spp,cur_spp_val);
              max_cur_spp = max(max_cur_spp,cur_spp_val);
              avg_cur_spp += cur_spp_val;
              num_cur_avg++;
              if (cur_spp_val < 10)
                num_cur_low++;
            }
          }
        }
        //std::cout << std::endl;
      }
      cur_spp->unmap();
      brdf->unmap();
      avg_cur_spp /= num_cur_avg;
      std::cout << "Num cur spp below 10spp" << num_cur_low << std::endl;
      std::cout << "Num theoretical spp below 10spp" << num_low << std::endl;
      std::cout << "Num current sampled" << num_cur_avg << std::endl;
      std::cout << "Num theoretical sampled" << num_avg << std::endl;

      std::cout << "Minimum SPP: " << min_cur_spp << std::endl;
      std::cout << "Maximum SPP: " << max_cur_spp << std::endl;
      std::cout << "Average SPP: " << avg_cur_spp << std::endl;
      std::cout << "Minimum Theoretical SPP: " << min_spp << std::endl;
      std::cout << "Maximum Theoretical SPP: " << max_spp << std::endl;
      std::cout << "Average Theoretical SPP: " << avg_spp << std::endl;
#else
      std::cout << "SPP stats turned off (GPU local buffer)" << std::endl;
#endif
      return true;
    }
  case 'C':
  case 'c':
    _sigma += 0.1;
    m_context["light_sigma"]->setFloat(_sigma);
    std::cout << "Light sigma is now: " << _sigma << std::endl;
    m_camera_changed = true;
    return true;
  case 'X':
  case 'x':
    _sigma -= 0.1;
    m_context["light_sigma"]->setFloat(_sigma);
    std::cout << "Light sigma is now: " << _sigma << std::endl;
    m_camera_changed = true;
    return true;

  case 'A':
  case 'a':
    std::cout << _frame_number << " frames." << std::endl;
    return true;
  case 'B':
  case 'b':
    _blur_occ = 1-_blur_occ;
    m_context["blur_occ"]->setUint(_blur_occ);
    if (_blur_occ)
      std::cout << "Blur: On" << std::endl;
    else
      std::cout << "Blur: Off" << std::endl;
    return true;

  case 'H':
  case 'h':
    _blur_wxf = 1-_blur_wxf;
    m_context["blur_wxf"]->setUint(_blur_wxf);
    if (_blur_wxf)
      std::cout << "Blur Omega x f: On" << std::endl;
    else
      std::cout << "Blur Omega x f: Off" << std::endl;
    return true;
  case 'E':
  case 'e':
    _err_vis = 1-_err_vis;
    m_context["err_vis"]->setUint(_err_vis);
    if (_err_vis)
      std::cout << "Err vis: On" << std::endl;
    else
      std::cout << "Err vis: Off" << std::endl;
    return true;
  case 'Z':
  case 'z':
    _view_mode = (_view_mode+1)%9;
    m_context["view_mode"]->setUint(_view_mode);
    switch(_view_mode) {
    case 0:
      std::cout << "View mode: Normal" << std::endl;
      break;
    case 1:
      std::cout << "View mode: Occlusion Only" << std::endl;
      break;
    case 2:
      std::cout << "View mode: Scale" << std::endl;
      break;
    case 3:
      std::cout << "View mode: Current SPP" << std::endl;
      break;
    case 4:
      std::cout << "View mode: Theoretical SPP" << std::endl;
      break;
    case 5:
      std::cout << "Use filter (normals)" << std::endl;
      break;
    case 6:
      std::cout << "Use filter (unoccluded)" << std::endl;
      break;
    case 7:
      std::cout << "View unconverged pixels" << std::endl;
      break;
    default:
      std::cout << "View mode: Unknown" << std::endl;
      break;
    }
    return true;

  case '\'':
    _lin_sep_blur = 1-_lin_sep_blur;
    m_context["lin_sep_blur"]->setUint(_lin_sep_blur);
    if (_lin_sep_blur)
      std::cout << "Linearly Separable Blur: On" << std::endl;
    else
      std::cout << "Linearly Separable Blur: Off" << std::endl;
    return true;
  case 'P':
  case 'p':
    _show_progressive = 1-_show_progressive;
    m_context["show_progressive"]->setUint(_show_progressive);
    if (_show_progressive)
      std::cout << "Blur progressive: On" << std::endl;
    else
      std::cout << "Blur progressive: Off" << std::endl;
    return true;
  case '.':
    if(_pixel_radius.x > 1000)
      return true;
    _pixel_radius += make_int2(1,1);
    m_context["pixel_radius"]->setInt(_pixel_radius);
    std::cout << "Pixel radius now: " << _pixel_radius.x << "," << _pixel_radius.y << std::endl;
    return true;
  case ',':
    if (_pixel_radius.x < 2)
      return true;
    _pixel_radius -= make_int2(1,1);
    m_context["pixel_radius"]->setInt(_pixel_radius);
    std::cout << "Pixel radius now: " << _pixel_radius.x << "," << _pixel_radius.y << std::endl;
    return true;
  case 'S':
  case 's':
    {
      std::stringstream fname;
      fname << "output_";
      fname << std::setw(7) << std::setfill('0') << output_num;
      fname << ".ppm";
      Buffer output_buf = _scene->getOutputBuffer();
      sutilDisplayFilePPM(fname.str().c_str(), output_buf->get());
      output_num++;
      std::cout << "Saved file" << std::endl;
      return true;
    }
  case 'Y':
  case 'y':

    return true;

    /*m_context["eye"]->setFloat( camera_data.eye );
    m_context["U"]->setFloat( camera_data.U );
    m_context["V"]->setFloat( camera_data.V );
    m_context["W"]->setFloat( camera_data.W ); */
  case 'R':
  case 'r':
    sutilCurrentTime(&_previous_frame_time);
    _anim_t = 0;
    return true;
  case 'T':
  case 't':
    _is_anim = 1-_is_anim;
    return true;
  }
  return false;

}

void appendGeomGroup(GeometryGroup& target, GeometryGroup& source)
{
  int ct_target = target->getChildCount();
  int ct_source = source->getChildCount();
  target->setChildCount(ct_target+ct_source);
  for(int i=0; i<ct_source; i++)
    target->setChild(ct_target + i, source->getChild(i));
}

#if SCENE==4
void Arealight::createGeometry()
  //spheres
{
  //Make some temp geomgroups
  GeometryGroup ground_geom_group = m_context->createGeometryGroup();
  GeometryGroup spheres_geom_group = m_context->createGeometryGroup();
  GeometryGroup spheres2_geom_group = m_context->createGeometryGroup();
  GeometryGroup cones_geom_group = m_context->createGeometryGroup();
  GeometryGroup torus_geom_group = m_context->createGeometryGroup();

  //Set some materials
  Material ground_mat = m_context->createMaterial();
  ground_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  ground_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  ground_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["Kd"]->setFloat( 0.9f, 0.8f, 0.4f );
  ground_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["phong_exp"]->setFloat( 100.0f );
  ground_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["obj_id"]->setInt(10);

  Material spheres_mat = m_context->createMaterial();
  spheres_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  spheres_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  spheres_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres_mat["Kd"]->setFloat( 0.85, 0.35, 0.04 );
  spheres_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres_mat["phong_exp"]->setFloat( 100.0f );
  spheres_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres_mat["obj_id"]->setInt(11);

  Material spheres2_mat = m_context->createMaterial();
  spheres2_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  spheres2_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  spheres2_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres2_mat["Kd"]->setFloat( 0.5, 0.15, 0.04 );
  spheres2_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres2_mat["phong_exp"]->setFloat( 100.0f );
  spheres2_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres2_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  spheres2_mat["obj_id"]->setInt(11);

  Material cones_mat = m_context->createMaterial();
  cones_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  cones_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  cones_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  cones_mat["Kd"]->setFloat( 0.04, 0.14, 0.5 );
  cones_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  cones_mat["phong_exp"]->setFloat( 100.0f );
  cones_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  cones_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  cones_mat["obj_id"]->setInt(12);

  Material torus_mat = m_context->createMaterial();
  torus_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  torus_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  torus_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  torus_mat["Kd"]->setFloat( 0.04, 0.5, 0.14 );
  torus_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  torus_mat["phong_exp"]->setFloat( 100.0f );
  torus_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  torus_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  torus_mat["obj_id"]->setInt(13);


  //Transformations
  Matrix4x4 overall_xform = Matrix4x4::scale(make_float3(0.1));

  Matrix4x4 ground_xform = overall_xform
    *Matrix4x4::rotate(14*M_PI/180.0, make_float3(1,0,0))
    *Matrix4x4::scale(make_float3(2,2,2))
    *Matrix4x4::translate(make_float3(7,-10,-39))
    *Matrix4x4::rotate(10*M_PI/180.0, make_float3(0,1,0));
  Matrix4x4 spheres_xform = overall_xform
    *Matrix4x4::rotate(M_PI*0.6,make_float3(0,1,0))
    *Matrix4x4::translate(make_float3(0,10,0));
  Matrix4x4 cones_xform = overall_xform
    *Matrix4x4::translate(make_float3(-40,10,40));
  Matrix4x4 torus_xform = overall_xform
    *Matrix4x4::translate(make_float3(10,30,40))
    *Matrix4x4::rotate(3*M_PI/2, make_float3(0,1,0))
    *Matrix4x4::rotate(M_PI/2, make_float3(1,0,0));
  Matrix4x4 spheres2_xform;

  cones_xform = spheres_xform;
  torus_xform = Matrix4x4::translate(make_float3(-1.0,1.7,0)) * spheres_xform;
  spheres2_xform = spheres_xform;

  //Load the OBJ's
  ObjLoader * ground_loader = new ObjLoader( texpath("spheres/ground.obj").c_str(), m_context, ground_geom_group, ground_mat );
  ground_loader->load(ground_xform);
  ObjLoader * spheres_loader = new ObjLoader( texpath("spheres/spheres3.obj").c_str(), m_context, spheres_geom_group, spheres_mat );
  spheres_loader->load(spheres_xform);
  //ObjLoader * spheres2_loader = new ObjLoader( texpath("spheres/spheres2.obj").c_str(), m_context, spheres2_geom_group, spheres2_mat );
  //spheres2_loader->load(spheres2_xform);
  ObjLoader * cones_loader = new ObjLoader( texpath("spheres/cones.obj").c_str(), m_context, cones_geom_group, cones_mat );
  cones_loader->load(cones_xform);
  ObjLoader * torus_loader = new ObjLoader( texpath("spheres/torus.obj").c_str(), m_context, torus_geom_group, torus_mat );
  torus_loader->load(torus_xform);

  //Make one big geom group
  GeometryGroup geom_group = m_context->createGeometryGroup();
  appendGeomGroup(geom_group, ground_geom_group);
  appendGeomGroup(geom_group, spheres_geom_group);
  //appendGeomGroup(geom_group, spheres2_geom_group);
  appendGeomGroup(geom_group, cones_geom_group);
  appendGeomGroup(geom_group, torus_geom_group);
  //appendGeomGroup(geom_group, rock_geom_group);

  //geom_group->setChild(ct, global);
  geom_group->setAcceleration( m_context->createAcceleration("Bvh", "Bvh") );

  //Set the geom group
  m_context["top_object"]->set( geom_group );
  m_context["top_shadower"]->set( geom_group );
}
#endif

#if SCENE==3
//tentacles
void Arealight::createGeometry()
{
  //Make some temp geomgroups
  GeometryGroup ground_geom_group = m_context->createGeometryGroup();
  GeometryGroup tentacles_geom_group = m_context->createGeometryGroup();
  GeometryGroup rock_geom_group = m_context->createGeometryGroup();

  //Set some materials
  Material ground_mat = m_context->createMaterial();
  ground_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  ground_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  ground_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["Kd"]->setFloat( 1,0.9,0.4 );
  ground_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["phong_exp"]->setFloat( 100.0f );
  ground_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["obj_id"]->setInt(10);

  Material tentacles_mat = m_context->createMaterial();
  tentacles_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  tentacles_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  tentacles_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  tentacles_mat["Kd"]->setFloat( 0.5, 0.15, 0.04 );
  tentacles_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  tentacles_mat["phong_exp"]->setFloat( 100.0f );
  tentacles_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  tentacles_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  tentacles_mat["obj_id"]->setInt(11);

  Material rock_mat = m_context->createMaterial();
  rock_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  rock_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  rock_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  rock_mat["Kd"]->setFloat( 0.7,0.7,0.7 );
  rock_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  rock_mat["phong_exp"]->setFloat( 100.0f );
  rock_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  rock_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  rock_mat["obj_id"]->setInt(12);


  //Transformations
  Matrix4x4 overall_xform = Matrix4x4::translate(make_float3(0,-5.0,0));

  Matrix4x4 ground_xform = overall_xform
    * Matrix4x4::translate(make_float3(5.5653,-8,-20.86169));

  Matrix4x4 ground_xform2 = overall_xform
    * Matrix4x4::translate(make_float3(-20,1,-10))
    * Matrix4x4::scale(make_float3(100,10,100));

  Matrix4x4 ground_xform3 = overall_xform
    * Matrix4x4::translate(make_float3(-20,-6,-10));
    //* Matrix4x4::scale(make_float3(0.5));

  Matrix4x4 rock_xform = overall_xform
    * Matrix4x4::translate(make_float3(-12.0,-0.5,-8.0))
    * Matrix4x4::rotate(110 * M_PI/180, make_float3(0,1,0))
    * Matrix4x4::scale(make_float3(3));


  Matrix4x4 tentacles_xform = overall_xform
    * Matrix4x4::translate(make_float3(-10,0,-2))
    * Matrix4x4::rotate(-10*M_PI/180, make_float3(0,0,1))
    * Matrix4x4::rotate(35*M_PI/180, make_float3(0,1,0))
    * Matrix4x4::scale(make_float3(2));
  
  tentacles_xform = optix::Matrix4x4::identity();

  //Load the OBJ's
  //ObjLoader * ground_loader = new ObjLoader( texpath("tentacles2/tentacles_on_wavyground.obj").c_str(), m_context, ground_geom_group, ground_mat );
  //ground_loader->load(ground_xform);
  //ObjLoader * ground_loader2 = new ObjLoader( texpath("grids2/floor.obj").c_str(), m_context, ground_geom_group, ground_mat );
  //ground_loader2->load(ground_xform2);
  ObjLoader * ground_loader3 = new ObjLoader( texpath("spheres/ground.obj").c_str(), m_context, ground_geom_group, ground_mat );
  ground_loader3->load(ground_xform3);
  ObjLoader * tentacles_loader = new ObjLoader( texpath("tentacles2/tentacles_tree1.obj").c_str(), m_context, tentacles_geom_group, tentacles_mat );
  tentacles_loader->load(tentacles_xform);
  ObjLoader * rock_loader = new ObjLoader( texpath("tentacles2/tentacles_rock2.obj").c_str(), m_context, rock_geom_group, rock_mat );
  rock_loader->load(rock_xform);


  _trans = m_context->createTransform();
  _trans->setMatrix( false, optix::Matrix4x4::identity().getData(), 0);
  _trans->setChild( tentacles_geom_group );


  //Make one big geom group
  GeometryGroup geom_group = m_context->createGeometryGroup();
  appendGeomGroup(geom_group, ground_geom_group);
  //appendGeomGroup(geom_group, tentacles_geom_group);
  //appendGeomGroup(geom_group, rock_geom_group);

  //geom_group->setChild(ct, global);
  geom_group->setAcceleration( m_context->createAcceleration("Bvh", "Bvh") );


  _top_grp = m_context->createGroup();
  _top_grp->setChildCount(2);
  _top_grp->setChild(0, geom_group);
  _top_grp->setChild(1, _trans);

  _top_grp->setAcceleration(m_context->createAcceleration("Bvh", "Bvh"));
  _top_grp->getAcceleration()->setProperty("refit","1");

  //Set the geom group
  m_context["top_object"]->set( _top_grp );
  m_context["top_shadower"]->set( _top_grp );
}
#endif

#if SCENE==1
//grids2
void Arealight::createGeometry()
{
  //Intersection programs
  Program closest_hit = m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3");
  Program any_hit = m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow");

  //Make some temp geomgroups
  _top_grp = m_context->createGroup();
  GeometryGroup floor_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid1_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid3_geom_group = m_context->createGeometryGroup();
  GeometryGroup grid2_geom_group = m_context->createGeometryGroup();

  //Set some materials
  Material floor_mat = m_context->createMaterial();
  floor_mat->setClosestHitProgram(0, closest_hit);
  floor_mat->setAnyHitProgram(1, any_hit);
  floor_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  floor_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["phong_exp"]->setFloat( 100.0f );
  floor_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  floor_mat["obj_id"]->setInt(10);

  Material grid1_mat = m_context->createMaterial();
  grid1_mat->setClosestHitProgram(0, closest_hit);
  grid1_mat->setAnyHitProgram(1, any_hit);
  grid1_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["Kd"]->setFloat( 0.72f, 0.100741f, 0.09848f );
  grid1_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["phong_exp"]->setFloat( 100.0f );
  grid1_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid1_mat["obj_id"]->setInt(11);

  Material grid2_mat = m_context->createMaterial();
  grid2_mat->setClosestHitProgram(0, closest_hit);
  grid2_mat->setAnyHitProgram(1, any_hit);
  grid2_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["Kd"]->setFloat( 0.0885402f, 0.77f, 0.08316f );
  grid2_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["phong_exp"]->setFloat( 100.0f );
  grid2_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid2_mat["obj_id"]->setInt(12);

  Material grid3_mat = m_context->createMaterial();
  grid3_mat->setClosestHitProgram(0, closest_hit);
  grid3_mat->setAnyHitProgram(1, any_hit);
  grid3_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["Kd"]->setFloat( 0.123915f, 0.192999f, 0.751f );
  grid3_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["phong_exp"]->setFloat( 100.0f );
  grid3_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  grid3_mat["obj_id"]->setInt(13);

  //Transformations
  Matrix4x4 floor_xform = Matrix4x4::identity();
  float *floor_xform_m = floor_xform.getData();
  floor_xform_m[0] = 4.0;
  floor_xform_m[10] = 4.0;
  floor_xform_m[12] = 0.0778942;
  floor_xform_m[14] = 0.17478;

  Matrix4x4 grid1_xform = Matrix4x4::identity();
  float *grid1_xform_m = grid1_xform.getData();
  grid1_xform_m[0] = 0.75840;
  grid1_xform_m[1] = 0.6232783;
  grid1_xform_m[2] = -0.156223;
  grid1_xform_m[3] = 0.0;
  grid1_xform_m[4] = -0.465828;
  grid1_xform_m[5] = 0.693876;
  grid1_xform_m[6] = 0.549127;
  grid1_xform_m[7] = 0.0;
  grid1_xform_m[8] = 0.455878;
  grid1_xform_m[9] = -0.343688;
  grid1_xform_m[10] = 0.821008;
  grid1_xform_m[11] = 0.0;
  grid1_xform_m[12] = 2.18526;
  grid1_xform_m[13] = 1.0795;
  grid1_xform_m[14] = 1.23179;
  grid1_xform_m[15] = 1.0;

  Matrix4x4 grid2_xform = Matrix4x4::identity();
  float *grid2_xform_m = grid2_xform.getData();
  grid2_xform_m[0] = 0.893628;
  grid2_xform_m[1] = 0.203204;
  grid2_xform_m[2] = -0.40017;
  grid2_xform_m[3] = 0.0;
  grid2_xform_m[4] = 0.105897;
  grid2_xform_m[5] = 0.770988;
  grid2_xform_m[6] = 0.627984;
  grid2_xform_m[7] = 0.0;
  grid2_xform_m[8] = 0.436135;
  grid2_xform_m[9] = -0.603561;
  grid2_xform_m[10] = 0.667458;
  grid2_xform_m[11] = 0.0;
  grid2_xform_m[12] = 0.142805;
  grid2_xform_m[13] = 1.0837;
  grid2_xform_m[14] = 0.288514;
  grid2_xform_m[15] = 1.0;



  Matrix4x4 grid3_xform = Matrix4x4::identity();
  float *grid3_xform_m = grid3_xform.getData();
  grid3_xform_m[0] = 0.109836;
  grid3_xform_m[1] = 0.392525;
  grid3_xform_m[2] = -0.913159;
  grid3_xform_m[3] = 0.0;
  grid3_xform_m[4] = 0.652392;
  grid3_xform_m[5] = 0.664651;
  grid3_xform_m[6] = 0.364174;
  grid3_xform_m[7] = 0.0;
  grid3_xform_m[8] = 0.74988;
  grid3_xform_m[9] = -0.635738;
  grid3_xform_m[10] = -0.183078;
  grid3_xform_m[11] = 0.0;
  grid3_xform_m[12] = -2.96444;
  grid3_xform_m[13] = 1.86879;
  grid3_xform_m[14] = 1.00696;
  grid3_xform_m[15] = 1.0;

  floor_xform = floor_xform.transpose();
  grid1_xform = grid1_xform.transpose();
  grid2_xform = grid2_xform.transpose();
  grid3_xform = grid3_xform.transpose();

  //Load the OBJ's
  ObjLoader * floor_loader = new ObjLoader( texpath("grids2/floor.obj").c_str(), m_context, floor_geom_group, floor_mat );
  floor_loader->load(floor_xform);
  ObjLoader * grid1_loader = new ObjLoader( texpath("grids2/grid1.obj").c_str(), m_context, grid1_geom_group, grid1_mat );
  grid1_loader->load(grid1_xform);
  ObjLoader * grid3_loader = new ObjLoader( texpath("grids2/grid3.obj").c_str(), m_context, grid3_geom_group, grid3_mat );
  grid3_loader->load(grid3_xform);
  ObjLoader * grid2_loader = new ObjLoader( texpath("grids2/grid2.obj").c_str(), m_context, grid2_geom_group, grid2_mat );
  grid2_loader->load(grid2_xform);



  //Make one big geom group
  GeometryGroup geom_group = m_context->createGeometryGroup();

  geom_group->setChildCount(0);
  //geom_group->setChild( 0, floor_geom_group->getChild(0) );
  //geom_group->setChild( 1, grid1_geom_group->getChild(0) );
  //geom_group->setChild( 2, grid2_geom_group->getChild(0) );
  //geom_group->setChild( 3, grid2_geom_group->getChild(0) );
  //geom_group->setChild( 3, grid3_geom_group->getChild(0) );

  //appendGeomGroup(geom_group, floor_geom_group);

  _trans = m_context->createTransform();
  _trans2 = m_context->createTransform();

  grid2_geom_group->getAcceleration()->setProperty("refit", "1");
  grid3_geom_group->getAcceleration()->setProperty("refit", "1");

  _anim_geom = grid2_geom_group->getChild(0)->getGeometry();
  _anim_geom_group = grid2_geom_group;
  

  optix::Matrix4x4 test = optix::Matrix4x4::translate(make_float3(0,0,0));
  _trans->setMatrix( false, test.getData(), 0);
  _trans->setChild( grid2_geom_group );

  _trans2->setMatrix( false, test.getData(), 0);
  _trans2->setChild( grid3_geom_group );

  //grid2_geom_group->getAcceleration()->markDirty();

  std::cout << "asdf" << std::endl;
  std::cout << geom_group->getChildCount() << std::endl;
  appendGeomGroup(geom_group, floor_geom_group);
  appendGeomGroup(geom_group, grid1_geom_group);
  //appendGeomGroup(geom_group, grid2_geom_group);
  //appendGeomGroup(geom_group, grid3_geom_group);
  //geom_group->setChild(ct, global);
  //geom_group->setAcceleration( m_context->createAcceleration("Sbvh", "Bvh") );
  geom_group->setAcceleration( m_context->createAcceleration("Bvh", "Bvh") );
  //geom_group->setAcceleration( m_context->createAcceleration("TriangleKdTree", "KdTree") );
  //Acceleration accl = floor_geom_group->getAcceleration();
  //std::cout << accl->getBuilder() << std::endl;
  //geom_group->setAcceleration( grid3_geom_group->getAcceleration() );

  //int ct = geom_group->getChildCount();
  //geom_group->setChildCount(ct+1);
  //geom_group->setChild(ct, _trans.get() );

  _top_grp->setChildCount(3);
  _top_grp->setChild(0, geom_group);
  _top_grp->setChild(1, _trans);
  _top_grp->setChild(2, _trans2);

  _top_grp->setAcceleration(m_context->createAcceleration("Bvh", "Bvh") );
  _top_grp->getAcceleration()->setProperty("refit", "1");

  //Set the geom group
  m_context["top_object"]->set( _top_grp );
  m_context["top_shadower"]->set( _top_grp );

  //m_context["top_object"]->set( grid2_geom_group );
  //m_context["top_shadower"]->set( grid2_geom_group );
}
#endif
#if SCENE==2
//balance
void Arealight::createGeometry()
{
  //Make some temp geomgroups
  GeometryGroup balance_geom_group = m_context->createGeometryGroup();
  GeometryGroup street_lamp_geom_group = m_context->createGeometryGroup();
  GeometryGroup street_lamp2_geom_group = m_context->createGeometryGroup();
  GeometryGroup tree_geom_group = m_context->createGeometryGroup();
  GeometryGroup tree2_geom_group = m_context->createGeometryGroup();

  //Set some materials
  Material balance_mat = m_context->createMaterial();
  balance_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  balance_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  balance_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  balance_mat["Kd"]->setFloat( 0.976f, 0.521f, 0.337f );
  balance_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  balance_mat["phong_exp"]->setFloat( 100.0f );
  balance_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  balance_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  balance_mat["obj_id"]->setInt(10);

  Material street_lamp_mat = m_context->createMaterial();
  street_lamp_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  street_lamp_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  street_lamp_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  street_lamp_mat["Kd"]->setFloat( 1.7f, 1.3f, 0.3f );
  street_lamp_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  street_lamp_mat["phong_exp"]->setFloat( 100.0f );
  street_lamp_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  street_lamp_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  street_lamp_mat["obj_id"]->setInt(11);

  Material tree_mat = m_context->createMaterial();
  tree_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  tree_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  tree_mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  tree_mat["Kd"]->setFloat( 0.500f, 0.380f, 0.320f );
  tree_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  tree_mat["phong_exp"]->setFloat( 100.0f );
  tree_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  tree_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  tree_mat["obj_id"]->setInt(12);

  Material ground_mat = m_context->createMaterial();
  ground_mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  ground_mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));
  //ground_mat["Ka"]->setFloat( 0.5f, 0.5f, 0.5f );
  ground_mat["Ka"]->setFloat( 0,0,0 );
  //ground_mat["Kd"]->setFloat( 0.620f, 0.770f, 0.650f );
  ground_mat["Kd"]->setFloat( 230.0/255.0,1,230.0/255.0 );
  ground_mat["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["phong_exp"]->setFloat( 100.0f );
  ground_mat["reflectivity"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["reflectivity_n"]->setFloat( 0.0f, 0.0f, 0.0f );
  ground_mat["obj_id"]->setInt(13);

  //Transformations
  Matrix4x4 overall_xform = Matrix4x4::translate(make_float3(-2.0f, 2.0f, -6.0))
    * Matrix4x4::rotate(-65.0f * M_PI/180.0f, make_float3(0.0f, 1.0f, 0.0f))
    * Matrix4x4::rotate(-15.0f * M_PI/180.0f, make_float3(0.0f, 1.0f, 0.0f))
    * Matrix4x4::scale(make_float3(12.0f,12.0f,12.0f));
  Matrix4x4 balance_xform = overall_xform 
    * Matrix4x4::translate(make_float3(0.0f, 0.015f, 0.0f));
  Matrix4x4 street_lamp_xform = overall_xform; 
  Matrix4x4 street_lamp2_xform = overall_xform 
    * Matrix4x4::rotate(M_PI,make_float3(0.0f, 1.0f, 0.0f));
  Matrix4x4 tree_xform = overall_xform * Matrix4x4::identity();
  Matrix4x4 tree2_xform = overall_xform 
    * Matrix4x4::rotate(M_PI,make_float3(0.0f, 1.0f, 0.0f))
    * Matrix4x4::translate(make_float3(0.5f, 0.0f, 0.5f))
    * Matrix4x4::rotate(-20.0f * M_PI/180.0f, make_float3(0.0f, 1.0f, 0.0f))
    * Matrix4x4::translate(make_float3(-0.5f, 0.0f, -0.5f))
    * Matrix4x4::translate(make_float3(0.0f, 0.01f, 0.0f));



  //Load the OBJ's
  ObjLoader * balance_loader = new ObjLoader( texpath("bench/Balance.obj").c_str(), m_context, balance_geom_group, balance_mat );
  balance_loader->load(balance_xform);
  ObjLoader * street_lamp_loader = new ObjLoader( texpath("bench/StreetLamp.obj").c_str(), m_context, street_lamp_geom_group, street_lamp_mat );
  street_lamp_loader->load(street_lamp_xform);
  ObjLoader * street_lamp2_loader = new ObjLoader( texpath("bench/StreetLamp.obj").c_str(), m_context, street_lamp2_geom_group, street_lamp_mat );
  street_lamp2_loader->load(street_lamp2_xform);
  ObjLoader * tree_loader = new ObjLoader( texpath("bench/Tree.obj").c_str(), m_context, tree_geom_group, tree_mat );
  tree_loader->load(tree_xform);
  ObjLoader * tree2_loader = new ObjLoader( texpath("bench/Tree.obj").c_str(), m_context, tree2_geom_group, tree_mat );
  tree2_loader->load(tree2_xform);

  // Floor geometry
  std::string pgram_ptx( ptxpath( "arealight", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( overall_xform * make_float4( -1.0f, 0.0f, -1.0f, 1.0f ) );
  float3 v1 = make_float3( overall_xform * make_float4( 2.0f, 0.0f, 0.0f, 0.0f ) );
  float3 v2 =  make_float3( overall_xform *make_float4( 0.0f, 0.0f, 2.0f, 0.0f ) );
  float3 normal = cross( v2, v1 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  //Make one big geom group
  GeometryGroup geom_group = m_context->createGeometryGroup();
  appendGeomGroup(geom_group, balance_geom_group);
  appendGeomGroup(geom_group, street_lamp_geom_group);
  appendGeomGroup(geom_group, street_lamp2_geom_group);
  appendGeomGroup(geom_group, tree_geom_group);
  appendGeomGroup(geom_group, tree2_geom_group);
  int ct = geom_group->getChildCount();

  geom_group->setChildCount( ct+1 );
  GeometryInstance ground = m_context->createGeometryInstance( parallelogram, &ground_mat, &ground_mat+1 );
  geom_group->setChild( ct, ground );
  //geom_group->setChild(ct, global);
  geom_group->setAcceleration( m_context->createAcceleration("Sbvh", "Bvh") );

  //Set the geom group
  m_context["top_object"]->set( geom_group );
  m_context["top_shadower"]->set( geom_group );
}
#endif

#if SCENE==0
void Arealight::createGeometry()
{
  std::string box_ptx( ptxpath( "arealight", "box.cu" ) ); 
  Program box_bounds = m_context->createProgramFromPTXFile( box_ptx, "box_bounds" );
  Program box_intersect = m_context->createProgramFromPTXFile( box_ptx, "box_intersect" );

  // Create box
  Geometry box = m_context->createGeometry();
  box->setPrimitiveCount( 1u );
  box->setBoundingBoxProgram( box_bounds );
  box->setIntersectionProgram( box_intersect );
  box["boxmin"]->setFloat( -2.0f, 0.0f, -2.0f );
  box["boxmax"]->setFloat(  2.0f, 7.0f,  2.0f );

  // Create chull
  Geometry chull = 0;

  // Sphere
  std::string sph_ptx( ptxpath( "arealight", "sphere.cu" ) ); 
  Program sph_bounds = m_context->createProgramFromPTXFile( sph_ptx, "bounds" );
  Program sph_intersect = m_context->createProgramFromPTXFile( sph_ptx, "intersect" );

  Geometry sphere = m_context->createGeometry();
  sphere->setBoundingBoxProgram( sph_bounds );
  sphere->setIntersectionProgram( sph_intersect );
  sphere->setPrimitiveCount( 1u );
  sphere["sphere"]->setFloat( 0, 5, 0, 4 );




  // Floor geometry
  std::string pgram_ptx( ptxpath( "arealight", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -16.0f, 0.01f, -16.0f );
  float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 32.0f );
  float3 normal = cross( v2, v1 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Materials
  std::string box_chname;
  box_chname = "closest_hit_radiance3";

  Material box_matl = m_context->createMaterial();
  Program box_ch = m_context->createProgramFromPTXFile( _ptx_path, box_chname );
  box_matl->setClosestHitProgram( 0, box_ch );
  Program box_ah = m_context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  box_matl->setAnyHitProgram( 1, box_ah );

  box_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  box_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  box_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  box_matl["phong_exp"]->setFloat( 88 );
  box_matl["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  box_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );
  box_matl["obj_id"]->setInt(1);


  std::string sph_chname;
  sph_chname = "closest_hit_radiance3";
  Material sph_matl = m_context->createMaterial();
  Program sph_ch = m_context->createProgramFromPTXFile( _ptx_path, sph_chname );
  sph_matl->setClosestHitProgram( 0, sph_ch );
  Program sph_ah = m_context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  sph_matl->setAnyHitProgram( 1, sph_ah );

  sph_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  sph_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  sph_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  sph_matl["phong_exp"]->setFloat( 88 );
  sph_matl["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  sph_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );
  sph_matl["obj_id"]->setInt(1);

  std::string floor_chname;
  floor_chname = "closest_hit_radiance3";
  Material floor_matl = m_context->createMaterial();
  Program floor_ch = m_context->createProgramFromPTXFile( _ptx_path, floor_chname );
  floor_matl->setClosestHitProgram( 0, floor_ch );

  Program floor_ah = m_context->createProgramFromPTXFile( _ptx_path, "any_hit_shadow" );
  floor_matl->setAnyHitProgram( 1, floor_ah );

  floor_matl["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  //floor_matl["Kd"]->setFloat( 194/255.f*.6f, 186/255.f*.6f, 151/255.f*.6f );
  //floor_matl["Ks"]->setFloat( 0.4f, 0.4f, 0.4f );
  floor_matl["Kd"]->setFloat( 1.0f, 1.0f, 1.0f );
  floor_matl["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  floor_matl["reflectivity"]->setFloat( 0.1f, 0.1f, 0.1f );
  floor_matl["reflectivity_n"]->setFloat( 0.05f, 0.05f, 0.05f );
  floor_matl["phong_exp"]->setFloat( 88 );
  floor_matl["tile_v0"]->setFloat( 0.25f, 0, .15f );
  floor_matl["tile_v1"]->setFloat( -.15f, 0, 0.25f );
  floor_matl["crack_color"]->setFloat( 0.1f, 0.1f, 0.1f );
  floor_matl["crack_width"]->setFloat( 0.02f );
  floor_matl["obj_id"]->setInt(2);


  // Glass material
  Material glass_matl;
  if( chull.get() ) {
    Program glass_ch = m_context->createProgramFromPTXFile( _ptx_path, "glass_closest_hit_radiance" );
    std::string glass_ahname;
    glass_ahname = "any_hit_shadow";
    Program glass_ah = m_context->createProgramFromPTXFile( _ptx_path, glass_ahname );
    glass_matl = m_context->createMaterial();
    glass_matl->setClosestHitProgram( 0, glass_ch );
    glass_matl->setAnyHitProgram( 1, glass_ah );

    glass_matl["importance_cutoff"]->setFloat( 1e-2f );
    glass_matl["cutoff_color"]->setFloat( 0.34f, 0.55f, 0.85f );
    glass_matl["fresnel_exponent"]->setFloat( 3.0f );
    glass_matl["fresnel_minimum"]->setFloat( 0.1f );
    glass_matl["fresnel_maximum"]->setFloat( 1.0f );
    glass_matl["refraction_index"]->setFloat( 1.4f );
    glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["refraction_maxdepth"]->setInt( 100 );
    glass_matl["reflection_maxdepth"]->setInt( 100 );
    float3 extinction = make_float3(.80f, .89f, .75f);
    glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
    glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );
  }


  // Create box
  Geometry lightParallelogram = m_context->createGeometry();
  lightParallelogram->setPrimitiveCount( 1u );
  lightParallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  lightParallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 lanchor = make_float3(0.0f, 15.0f, -16.0f);
  float3 lv1 = make_float3(0.0f, 10.0f, -16.0f);
  float3 lv2 = make_float3(10.0f, 15.0f, -16.0f);
  float3 lnormal = cross( v2, v1 );

  lnormal = normalize( lnormal );
  float ld = dot( lnormal, lanchor );
  lv1 *= 1.0f/dot( lv1, lv1 );
  lv2 *= 1.0f/dot( lv2, lv2 );
  float4 lplane = make_float4( lnormal, ld );
  lightParallelogram["plane"]->setFloat( lplane );
  lightParallelogram["v1"]->setFloat( lv1 );
  lightParallelogram["v2"]->setFloat( lv2 );
  lightParallelogram["anchor"]->setFloat( lanchor );


  //Load OBJ
  geomgroup = m_context->createGeometryGroup();
  geomgroup2 = m_context->createGeometryGroup();
  ObjLoader* loader = 0;
  std::string obj_file;
  obj_file = "kscene1.obj";
  //obj_file = "house2.obj";
  //obj_file = "cherrytree2.obj";
  //obj_file = "cherrytree.obj";
  //obj_file = "plant.obj";
  //obj_file = "bunny.obj";
  //obj_file = "sphere.obj";

  std::string obj_file2;
  //obj_file2 = "house2.obj";
  obj_file2 = "cherrytree2.obj";

  //just for an ex

  //Material
  Material mat = m_context->createMaterial();
  mat->setClosestHitProgram(0, m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_radiance3"));
  mat->setAnyHitProgram(1, m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow"));

  mat["Ka"]->setFloat( 0.0f, 0.0f, 0.0f );
  mat["Kd"]->setFloat( .40f, 0.8f, .23f );
  mat["Ks"]->setFloat( 1.0f, 1.0f, 1.0f );
  mat["phong_exp"]->setFloat( 88 );
  mat["reflectivity"]->setFloat( 0.05f, 0.05f, 0.05f );
  mat["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );  
  mat["obj_id"]->setInt(10);

  //Matrix4x4 obj_xform = Matrix4x4::identity()* Matrix4x4::translate(make_float3(0,3,0)) * Matrix4x4::rotate(0, make_float3(1,0,0)) * Matrix4x4::scale(make_float3(3.0));
  Matrix4x4 obj_xform = Matrix4x4::identity() * Matrix4x4::scale(make_float3(10.0));
  Matrix4x4 obj_xform2 = Matrix4x4::identity() * Matrix4x4::translate(make_float3(4.0,0.0,-5.0)) * Matrix4x4::scale(make_float3(1.0));

  loader = new ObjLoader( texpath(obj_file).c_str(), m_context, geomgroup, mat );
  loader->load(obj_xform);  

  //ObjLoader * loader2 = new ObjLoader( texpath(obj_file2).c_str(), m_context, geomgroup2, mat );
  //loader2->load(obj_xform2);  

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( box, &box_matl, &box_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl, &floor_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( sphere, &sph_matl, &sph_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( lightParallelogram, &box_matl, &box_matl+1 ) );

  if(chull.get())
    gis.push_back( m_context->createGeometryInstance( chull, &glass_matl, &glass_matl+1 ) );

  // Place all in group
  int ct = geomgroup->getChildCount();
  geomgroup->setChildCount( ct + 1 );
  geomgroup->setChild( ct, gis[1] );
  ct = geomgroup->getChildCount();
  //hacky way to add a second object, clean up later
  /*
  int ct2 = geomgroup2->getChildCount();
  geomgroup->setChildCount( ct + ct2 );
  for(int i=0; i<ct2; i++)
  geomgroup->setChild( ct+i, geomgroup2->getChild(i) );
  //geomgroup->setChild( ct+1, gis[3] );
  */

  /*
  GeometryGroup shadowergroup = m_context->createGeometryGroup();
  shadowergroup->setChildCount( ct + 1 );
  for(int i=0; i<ct+1; i++) {
  shadowergroup->setChild(i,geomgroup->getChild(i));
  }

  shadowergroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  */


  m_context["top_object"]->set( geomgroup );
  m_context["top_shadower"]->set( geomgroup );




  /*
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setChild( 2, gis[2] );
  if(chull.get())
  geometrygroup->setChild( 3, gis[3] );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
  */
}
#endif


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -t  | --texture-path <path>                Specify path to texture directory\n"
    << "        --dim=<width>x<height>               Set image dimensions\n"
    << std::endl;

  std::cout
    << "Key bindings:" << std::endl
    << "c/x: Increase/decrease light sigma" << std::endl
    << "a: Current frame number" << std::endl
    << "b: Toggle filtering" << std::endl
    << "e: Toggle error visualization" << std::endl
    << "z: Toggle Zmin view" << std::endl

    // This stuff is hardcoded for now
    //<< "\\ Toggle Linearly Separable Blur" << std::endl
    //<< "p: Toggle Progressive Blur" << std::endl

    << "./,: Increase/decrease pixel radius" << std::endl
    << "v: Output SPP stats" << std::endl
    << "m: Toggle BRDF display" << std::endl
    << "n: Toggle Occlusion" << std::endl
    << "u/j, i/k, o/l: Increase/decrease light in x,y,z" << std::endl
    << "y: Output camera info" << std::endl
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  unsigned int width = 1080u, height = 720u;
  //unsigned int width = 1600u, height = 1080u;
  //unsigned int width = 640u, height = 480u;

  std::string texture_path;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg.substr( 0, 6 ) == "--dim=" ) {
      std::string dims_arg = arg.substr(6);
      if ( sutilParseImageDimensions( dims_arg.c_str(), &width, &height ) != RT_SUCCESS ) {
        std::cerr << "Invalid window dimensions: '" << dims_arg << "'" << std::endl;
        printUsageAndExit( argv[0] );
      }
    } else if ( arg == "-t" || arg == "--texture-path" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      texture_path = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( texture_path.empty() ) {
    texture_path = std::string( sutilSamplesDir() ) + "/arealight_research/data";
  }


  std::stringstream title;
  title << "arealight";
  try {
    _scene = new Arealight(texture_path);
    _scene->setDimensions( width, height );
    //dont time out progressive
    GLUTDisplay::setProgressiveDrawingTimeout(0.0);
    GLUTDisplay::run( title.str(), _scene, GLUTDisplay::CDProgressive ); //GLUTDisplay::CDNone );//GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
