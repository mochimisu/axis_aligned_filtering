/*
* filtergi.cpp
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

using namespace optix;
unsigned int width = 1080u, height = 720u;
float fov = 35.f;

class FilterGI : public SampleScene
{
public:
  FilterGI(const std::string& texture_path)
    : SampleScene(), _width(width), _height(height), texture_path( texture_path )
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

  GeometryInstance createParallelogram( const float3& anchor,
                                        const float3& offset1,
                                        const float3& offset2);

  GeometryInstance createLightParallelogram( const float3& anchor,
      const float3& offset1,
      const float3& offset2,
      int lgt_instance = -1);
  void setMaterial( GeometryInstance& gi,
      Material material,
      const std::string& color_name,
      const float3& color);



  virtual bool   keyPressed(unsigned char key, int x, int y);

private:
  std::string texpath( const std::string& base );
  void resetAccumulation();
  void createGeometry();

  bool _accel_cache_loaded;
  Program        m_pgram_bounding_box;
  Program        m_pgram_intersection;

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

  uint _filter_indirect;
  uint _blur_wxf;
  uint _err_vis;
  int _view_mode;
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

  BasicLight * _env_lights;
  uint _show_brdf;
  uint _show_occ;

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

FilterGI* _scene;
int output_num = 0;

uint max_spp = 15;//250;

void FilterGI::initScene( InitialCameraData& camera_data )
{
  _anim_t = 0;
  _total_avg_cur_spp = 0;
  sutilCurrentTime(&_previous_frame_time);
  _is_anim = true;
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "filtergi.cu";
  _ptx_path = ptxpath( "filtergi", ss.str() );

  // context 
  m_context->setRayTypeCount( 3 );
  m_context->setEntryPointCount( 7 );
  m_context->setStackSize( 8000 );

  m_context["max_depth"]->setInt(100);
  m_context["direct_ray_type"]->setUint(0);
  m_context["indirect_ray_type"]->setUint(1);
  m_context["shadow_ray_type"]->setUint(2);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["importance_cutoff"]->setFloat( 0.01f );
  m_context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );
  m_context["max_spp"]->setUint(max_spp);
  m_context["image_dim"]->setUint(width, height);
  m_context["fov"]->setFloat(fov);

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, _width, _height) );

  //[bmw] my stuff
  //seed rng
  //(i have no idea if this is right)
  //fix size later
  Buffer indirect_rng_seeds = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, _width, _height);
  m_context["indirect_rng_seeds"]->set(indirect_rng_seeds);
  uint2* seeds = reinterpret_cast<uint2*>( indirect_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  indirect_rng_seeds->unmap();


  // new gi stuff
  Buffer direct_illum = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["direct_illum"]->set(direct_illum);

  Buffer indirect_illum = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["indirect_illum"]->set(indirect_illum);

  Buffer image_Kd = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["image_Kd"]->set(image_Kd);
  Buffer image_Ks = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["image_Ks"]->set(image_Ks);
  Buffer image_phong_exp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height);
  m_context["image_phong_exp"]->set(image_phong_exp);
  Buffer image_normal = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["image_normal"]->set(image_normal);

  Buffer indirect_illum_sep = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height*4);
  m_context["indirect_illum_sep"]->set(indirect_illum_sep);
  
  // 4 for 4 splits of hemisphere
  Buffer indirect_illum_accum = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, _width, _height*4);
  m_context["indirect_illum_accum"]->set(indirect_illum_accum);

  Buffer indirect_illum_blur1d = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height*4);
  m_context["indirect_illum_blur1d"]->set(indirect_illum_blur1d);

  Buffer indirect_illum_filt = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height*4);
  m_context["indirect_illum_filt"]->set(indirect_illum_filt);

  Buffer indirect_illum_filt_int = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height);
  m_context["indirect_illum_filt_int"]->set(indirect_illum_filt_int);

  Buffer zpmin = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height*4);
  m_context["z_perp"]->set(zpmin);
  Buffer zpminf = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height*4);
  m_context["z_perp_filter1d"]->set(zpminf);
  Buffer zpmino = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT2, _width, _height*4);
  m_context["z_perp_orig"]->set(zpmino);
  Buffer depthb = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height);
  m_context["depth"]->set(depthb);

  Buffer indirect_spp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, _width, _height*4);
  m_context["indirect_spp"]->set(indirect_spp);

  Buffer target_indirect_spp = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, _width, _height*4);
  m_context["target_indirect_spp"]->set(target_indirect_spp);

  Buffer use_filter = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_BYTE, _width, _height);
  m_context["use_filter"]->set(use_filter);

  Buffer omega_v_max = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height);
  m_context["omega_v_max"]->set(omega_v_max);

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
  }
   
  gauss_lookup->unmap();

  // world space buffer
  Buffer world_loc = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["world_loc"]->set( world_loc );

  Buffer n = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["n"]->set( n );

  Buffer direct_l = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["direct_l"]->set( direct_l );

  Buffer indirect_l = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT3, _width, _height );
  m_context["indirect_l"]->set( indirect_l );

  Buffer zmin = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT, _width, _height );
  m_context["zmin"]->set( zmin );

  _filter_indirect = 1;
  m_context["filter_indirect"]->setUint(_filter_indirect);

  _view_mode = 0;
  m_context["view_mode"]->setUint(_view_mode);


  _normal_rpp = 3;
  _brute_rpp = 2000;
  _max_rpp_pass = 8;
  float spp_mu = 2;

  m_context["normal_rpp"]->setUint(_normal_rpp);
  m_context["brute_rpp"]->setUint(_brute_rpp);
  m_context["max_rpp_pass"]->setUint(_max_rpp_pass);

  _pixel_radius = make_int2(10,10);
  m_context["pixel_radius"]->setInt(_pixel_radius);
  
  // Initial ampling program
  std::string camera_name;
  camera_name = "pinhole_camera_initial_sample";

  Program ray_gen_program = m_context->createProgramFromPTXFile( _ptx_path, camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Occlusion Filter programs
  std::string first_pass_indirect_filter_name = "indirect_filter_first_pass";
  Program first_indirect_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    first_pass_indirect_filter_name );
  m_context->setRayGenerationProgram( 2, first_indirect_filter_program );
  std::string second_pass_indirect_filter_name = "indirect_filter_second_pass";
  Program second_indirect_filter_program = m_context->createProgramFromPTXFile( _ptx_path, 
    second_pass_indirect_filter_name );
  m_context->setRayGenerationProgram( 3, second_indirect_filter_program );

  // second sampling
  std::string continued_sample_camera;
  continued_sample_camera = "pinhole_camera_continued_sample";

  Program continued_sample_program = m_context->createProgramFromPTXFile( _ptx_path, continued_sample_camera );
  m_context->setRayGenerationProgram( 4, continued_sample_program );


// second sampling
Program zf_fp = m_context->createProgramFromPTXFile( _ptx_path, "z_filter_first_pass" );
m_context->setRayGenerationProgram( 5, zf_fp );
Program zf_sp = m_context->createProgramFromPTXFile( _ptx_path, "z_filter_second_pass" );
m_context->setRayGenerationProgram( 6, zf_sp );

  // Display program
  std::string display_name;
  display_name = "display_camera";

  Program display_program = m_context->createProgramFromPTXFile( _ptx_path, display_name );
  m_context->setRayGenerationProgram( 1, display_program );

  // Exception / miss programs
  Program exception_program = m_context->createProgramFromPTXFile( _ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

  std::string miss_name;
  miss_name = "miss";
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( _ptx_path, miss_name ) );
  const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
  m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );


  float3 pos1 = make_float3(1.5, 16, 8);
  float3 pos2 = make_float3(-4.5, 21.8284, 3.8284);

  //float3 pos   = make_float3( 343.0f, 548.6f, 227.0f);
  float3 pos   = make_float3( 343.0f, 548.6f, 227.0f);

  BasicLight lights[] = {
    { pos,
      make_float3(15.f, 15.f, 5.f),
      1
    }
  };


  _env_lights = lights;
  light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);


  // Set up camera

    //sponza
  //camera_data = InitialCameraData( make_float3( 652.5f, 693.5f, 0.f ), // eye
  //make_float3( 614.0f, 654.0f, 0.0f ),    // lookat
  camera_data = InitialCameraData( make_float3( -542.f, 520.f, 162.f ), // eye
  make_float3( 166.f, 202.f, 251.f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      35.0f );                                // vfov
  /* cornell box
    camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -800.0f ), // eye
                                     make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
                                     make_float3( 0.0f, 1.0f,  0.0f ),       // up
                                     35.0f );                                // vfov
                                     */

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );


//#endif

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


Buffer FilterGI::getOutputBuffer()
{

  return m_context["output_buffer"]->getBuffer();
}

void FilterGI::trace( const RayGenCameraData& camera_data )
{
  _frame_number ++;


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
    _anim_t += 0.7 * time_elapsed; //0.6 * time_elapsed;
  float3 eye, u, v, w;
  eye.x = (float) (camera_data.eye.x * sin(_anim_t));
  eye.y = (float)( 0.2 + camera_data.eye.y + cos( _anim_t*1.5 ) );
  eye.z = (float)( 0.5+camera_data.eye.z*cos( _anim_t ) );
  float3 lookat = make_float3(0);


  PinholeCamera pc( eye, lookat, make_float3(0,1,0), fov, fov/(width/height) );
  pc.getEyeUVW( eye, u, v, w );
  m_context["eye"]->setFloat( eye );
  m_context["U"]->setFloat( u );
  m_context["V"]->setFloat( v );
  m_context["W"]->setFloat( w );
  //m_camera_changed = true;


//#define MOVE_CAMERA
#ifndef MOVE_CAMERA
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
#endif

  // do i need to reseed?
  Buffer indirect_rng_seeds = m_context["indirect_rng_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( indirect_rng_seeds->map() );
  for(unsigned int i = 0; i < _width * _height; ++i )
    seeds[i] = random2u();
  indirect_rng_seeds->unmap();

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );
  m_context["frame"]->setUint( _frame_number );

  int num_resample = ceil((float)_brute_rpp * _brute_rpp / (_max_rpp_pass * _max_rpp_pass));

  //Initial 16 Samples
  if (_frame_number == 0)
  {
    m_context->launch( 0, static_cast<unsigned int>(buffer_width),
      static_cast<unsigned int>(buffer_height) );
  }
  else
  {
  m_context->launch( 5, static_cast<unsigned int>(buffer_width),
  static_cast<unsigned int>(buffer_height*4) );
  m_context->launch( 6, static_cast<unsigned int>(buffer_width),
  static_cast<unsigned int>(buffer_height*4) );
	  m_context->launch( 4, static_cast<unsigned int>(buffer_width),
      static_cast<unsigned int>(buffer_height) );
  }
  //filter indirect
  m_context->launch( 2, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );
  m_context->launch( 3, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );

  m_context->launch( 1, static_cast<unsigned int>(buffer_width),
    static_cast<unsigned int>(buffer_height) );

}


void FilterGI::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}

std::string FilterGI::texpath( const std::string& base )
{
  return texture_path + "/" + base;
}

float4 make_plane( float3 n, float3 p )
{
  n = normalize(n);
  float d = -dot(n, p);
  return make_float4( n, d );
}

void FilterGI::resetAccumulation()
{
  _frame_number = 0;
  m_context["frame"]->setUint( _frame_number );
  _converged = false;

}


bool FilterGI::keyPressed(unsigned char key, int x, int y) {
  float delta = 0.5f;
  const int num_view_modes = 22;

  Buffer spp;
  switch(key) {
  case 'U':
  case 'u':
    {
      float3 d = make_float3(delta,0,0);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos += d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'J':
  case 'j':
    {
      float3 d = make_float3(delta,0,0);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos -= d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;

      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'I':
  case 'i':
    {
      float3 d = make_float3(0,delta,0);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos += d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;

      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'K':
  case 'k':
    {
      float3 d = make_float3(0,delta,0);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos -= d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;

      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }

  case 'O':
  case 'o':
    {
      float3 d = make_float3(0,0,delta);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos += d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;
      light_buffer->unmap();

      m_camera_changed = true;
      return true;
    }
  case 'L':
  case 'l':
    {
      float3 d = make_float3(0,0,delta);
      BasicLight* lights = reinterpret_cast<BasicLight*>(light_buffer->map());
      lights[0].pos -= d;

      std::cout << "light now at (" 
        << "," << lights[0].pos.x 
        << "," << lights[0].pos.y 
        << "," << lights[0].pos.z
        << ")" << std::endl;

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
      std::cout << "SPP stats" << std:: endl;
      Buffer spp = m_context["indirect_spp"]->getBuffer();
      Buffer target_spp = m_context["target_indirect_spp"]->getBuffer();
      Buffer valid = m_context["use_filter"]->getBuffer();
      int min_spp = 100000000.;
      int max_spp = 0;
      float avg_spp = 0;
      int target_min_spp = 100000000.;
      int target_max_spp = 0;
      float target_avg_spp = 0;
      float num_avg = 0;
      float target_num_avg = 0;

      int* spp_arr = reinterpret_cast<int*>( spp->map() );
      int* target_spp_arr = reinterpret_cast<int*>( target_spp->map() );
      //char* valid_arr = reinterpret_cast<char*>( valid->map() );

      for(unsigned int j = 0; j < _height*4; ++j ) {
        for(unsigned int i = 0; i < _width; ++i ) {
          //if (valid_arr[i+j*_width] > -1) {
            float cur_spp_val = spp_arr[i+j*_width];
            float cur_target_spp_val = target_spp_arr[i+j*_width];
            if (cur_spp_val > -0.001) {
              min_spp = min(min_spp,cur_spp_val);
              max_spp = max(max_spp,cur_spp_val);
              avg_spp += cur_spp_val;
              num_avg+= 0.25;
            }
            if (cur_target_spp_val > -0.001) {
              target_min_spp = min(target_min_spp, cur_target_spp_val);
              target_max_spp = max(target_max_spp, cur_target_spp_val);
              target_avg_spp += cur_target_spp_val;
              target_num_avg+=0.25;
            }
          //} 
        }
      }
      spp->unmap();
      target_spp->unmap();
      //valid->unmap();
      avg_spp /= num_avg;
      target_avg_spp /= target_num_avg;
      std::cout << "Minimum SPP: " << min_spp << std::endl;
      std::cout << "Maximum SPP: " << max_spp << std::endl;
      std::cout << "Average SPP: " << avg_spp << std::endl;
      std::cout << "Minimum Target SPP: " << target_min_spp << std::endl;
      std::cout << "Maximum Target SPP: " << target_max_spp << std::endl;
      std::cout << "Average Target SPP: " << target_avg_spp << std::endl;
      return true;
    }


  case 'A':
  case 'a':
    std::cout << _frame_number << " frames." << std::endl;
    return true;
  case 'B':
  case 'b':
    _filter_indirect = 1-_filter_indirect;
    m_context["filter_indirect"]->setUint(_filter_indirect);
    if (_filter_indirect)
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
    if (key == 'Z')
      _view_mode = (_view_mode-1)%num_view_modes;
    else
      _view_mode = (_view_mode+1)%num_view_modes;
    if (_view_mode < 0)
      _view_mode += num_view_modes;
    m_context["view_mode"]->setUint(_view_mode);
    switch(_view_mode) {
    case 0:
      std::cout << "View mode: Normal" << std::endl;
      break;
    case 1:
      std::cout << "View mode: Direct Only" << std::endl;
      break;
    case 2:
      std::cout << "View mode: Indirect only" << std::endl;
      break;
    case 3:
      std::cout << "View mode: Omega_v_max" << std::endl;
      break;
    case 4:
      std::cout << "View mode: Pixels using filter" << std::endl;
      break;
    case 5:
    case 6:
    case 7:
    case 8:
      std::cout << "View mode: Indirect Bucket " << (_view_mode-5) << std::endl;
      break;
    case 9:
    case 10:
    case 11:
    case 12:
      std::cout << "View mode: Z Perpendicular (min) bucket " << (_view_mode-9) << std::endl;
      break;
    case 13:
      std::cout << "View mode: converged pixels" << std::endl;
      break;
      
    case 14:
    case 15:
    case 16:
    case 17:
      std::cout << "View mode: Target SPP bucket " << (_view_mode-14) << std::endl;
      break;


    case 18:
    case 19:
    case 20:
    case 21:
      std::cout << "View mode: SPP bucket " << (_view_mode-18) << std::endl;
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

GeometryInstance FilterGI::createParallelogram( const float3& anchor,
    const float3& offset1,
    const float3& offset2)
{
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d = dot( normal, anchor );
  float4 plane = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );

  GeometryInstance gi = m_context->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}
GeometryInstance FilterGI::createLightParallelogram( const float3& anchor,
    const float3& offset1,
    const float3& offset2,
    int lgt_instance)
{
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d = dot( normal, anchor );
  float4 plane = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["lgt_instance"]->setInt( lgt_instance );

  GeometryInstance gi = m_context->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}

void FilterGI::setMaterial( GeometryInstance& gi,
    Material material,
    const std::string& color_name,
    const float3& color)
{
  gi->addMaterial(material);
  gi[color_name]->setFloat(color);
}

void FilterGI::createGeometry()
{
  //Intersection programs
  Program closest_hit = m_context->createProgramFromPTXFile(_ptx_path, "closest_hit_direct");
  Program any_hit = m_context->createProgramFromPTXFile(_ptx_path, "any_hit_indirect");
  Program shadow_hit = m_context->createProgramFromPTXFile(_ptx_path, "any_hit_shadow");

  Program diffuse_ch = closest_hit;
  Program diffuse_ah = any_hit;

  //sponza
  BasicLight light;
  //light.pos = make_float3( 580.f, 680.f, 0.f);
  light.pos = make_float3( 580.f, 500.f, 0.f);
    Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( BasicLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer );

  _top_grp = m_context->createGroup();
  GeometryGroup sponza_geom_group = m_context->createGeometryGroup();

  Material floor_mat = m_context->createMaterial();
  floor_mat->setClosestHitProgram(0, closest_hit);
  floor_mat->setClosestHitProgram(1, any_hit);
  floor_mat["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  floor_mat["obj_id"]->setInt(10);

  //ObjLoader * floor_loader = new ObjLoader( texpath("dabrovic-sponza/sponza.obj").c_str(), m_context, sponza_geom_group, floor_mat );
  //ObjLoader * floor_loader = new ObjLoader( texpath("crytek_sponza/sponza.obj").c_str(), m_context, sponza_geom_group, floor_mat, true );
  ObjLoader * floor_loader = new ObjLoader( texpath("conference/conference.obj").c_str(), m_context, sponza_geom_group, floor_mat, true );
  //floor_loader->load();
  floor_loader->load();


  _top_grp->setChildCount(1);
  _top_grp->setChild(0, sponza_geom_group);

  _top_grp->setAcceleration(m_context->createAcceleration("Bvh", "Bvh") );
  _top_grp->getAcceleration()->setProperty("refit", "1");

  
  m_context["top_object"]->set( _top_grp );
  m_context["top_shadower"]->set( _top_grp );
}
//#endif

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
    texture_path = std::string( sutilSamplesDir() ) + "/gi_research/data";
  }


  std::stringstream title;
  title << "aafiltering gi";
  try {
    _scene = new FilterGI(texture_path);
    _scene->setDimensions( width, height );
    //dont time out progressive
    GLUTDisplay::setProgressiveDrawingTimeout(0.0);
    //GLUTDisplay::setUseSRGB(true);
    GLUTDisplay::run( title.str(), _scene, GLUTDisplay::CDProgressive ); //GLUTDisplay::CDNone );//GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
