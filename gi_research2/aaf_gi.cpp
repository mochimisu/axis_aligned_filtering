//------------------------------------------------------------------------------
//
// aaf_gi.cpp: render cornell box using path tracing.
//
//------------------------------------------------------------------------------

//=== Configuration

// Enable debug buffers (for stats and additional views)
#define DEBUG_BUF

// Choose scene:
// 0: Cornell box
// 1: Conference
// 2: Sponza
// 3: Cornell box 2 (Soham's w/ objs)
// 4: Cornell box 3 (Glossy defined by obj)
#define SCENE 4

//number of buckets to split hemisphere into
#define NUM_BUCKETS 1

//number of maximum samples per pixel
//#define MAX_SPP 100
#define MAX_SPP 25

//depth of indirect bounces
#define INDIRECT_BOUNCES 1

//enable GT mode
//#define GROUNDTRUTH

//64 = 4096 samples
#define GT_SAMPLES_SQRT 64

//default width, height
#define WIDTH 512u
#define HEIGHT 512u
//#define WIDTH 1024u
//#define HEIGHT 1024u

//=== End config



#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <GLUTDisplay.h>
#include <PPMLoader.h>
#include <ImageLoader.h>
#include <sampleConfig.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ObjLoader.h>

#include "random.h"
#include "aaf_gi.h"
#include "helpers.h"

using namespace optix;

//-----------------------------------------------------------------------------
//
// Helpers
//
//-----------------------------------------------------------------------------

namespace {
  std::string ptxpath( const std::string& base )
  {
    return std::string(sutilSamplesPtxDir()) + "/aaf_gi_generated_" + 
      base + ".ptx";
  }
}

//-----------------------------------------------------------------------------
//
// GIScene
//
//-----------------------------------------------------------------------------

class GIScene: public SampleScene
{
public:
  // Set the actual render parameters below in main().
  GIScene()
  : m_rr_begin_depth(1u)
  , m_max_depth(100u)
  , m_sqrt_num_samples( 0u )
  , m_width(512u)
  , m_height(512u)
  {}

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void   setNumSamples( unsigned int sns ) { 
    m_sqrt_num_samples= sns; 
  }
  void   setDimensions( const unsigned int w, const unsigned int h ) { 
    m_width = w; m_height = h; 
  }

private:
  // Should return true if key was handled, false otherwise.
  virtual bool keyPressed(unsigned char key, int x, int y);
  void createSceneCornell(InitialCameraData& camera_data);
  void createSceneCornell2(InitialCameraData& camera_data);
  void createSceneCornell3(InitialCameraData& camera_data);
  void createSceneSponza(InitialCameraData& camera_data);
  void createSceneConference(InitialCameraData& camera_data);

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

  Program        m_pgram_bounding_box;
  Program        m_pgram_intersection;

  unsigned int   m_rr_begin_depth;
  unsigned int   m_max_depth;
  unsigned int   m_sqrt_num_samples;
  unsigned int   m_width;
  unsigned int   m_height;
  unsigned int   m_frame;
  unsigned int   m_sampling_strategy;

  //AAF GI
  unsigned int m_view_mode;
  unsigned int m_first_pass_spb_sqrt;
  unsigned int m_brute_spb;
  unsigned int m_max_spb_pass;
  unsigned int m_use_textures;

  float m_max_heatmap_val;

  bool m_filter_indirect;
  bool m_prefilter_indirect;
  bool m_filter_z;

#ifdef GROUNDTRUTH
  int m_gt_total_pass;
#endif
};

GIScene scene;
int output_num = 0;
const unsigned int num_buckets = NUM_BUCKETS;

void GIScene::initScene( InitialCameraData& camera_data )
{
  m_context->setRayTypeCount( 5 );
  m_context->setEntryPointCount( 10 );
  m_context->setStackSize( 1800 );

  m_context["scene_epsilon"]->setFloat( 0.01f );
  m_context["max_depth"]->setUint(m_max_depth);
  m_context["pathtrace_shadow_ray_type"]->setUint(1u);
  m_context["pathtrace_bsdf_shadow_ray_type"]->setUint(2u);
  m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);


  // Setup output buffer
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, m_width, m_height );
  output_buffer->set(buffer);



  m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3(0.0f) );

  // Setup programs
  std::string ptx_path = ptxpath( "aaf_gi", "aaf_gi.cu" );
  Program exception_program = m_context->createProgramFromPTXFile( ptx_path, 
      "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, 
        "miss" ) );

  m_context["frame_number"]->setUint(1);

   // Index of sampling_stategy (BSDF, light, MIS)
  m_sampling_strategy = 0;
  m_context["sampling_stategy"]->setInt(m_sampling_strategy);

  // AAF programs
  Program direct_z_sample_prog = m_context->createProgramFromPTXFile(
      ptx_path, "sample_direct_z");
  Program z_filt_first_prog = m_context->createProgramFromPTXFile(
      ptx_path, "z_filter_first_pass");
  Program z_filt_second_prog = m_context->createProgramFromPTXFile(
      ptx_path, "z_filter_second_pass");
  Program sample_indirect_prog = m_context->createProgramFromPTXFile(
      ptx_path, "sample_indirect");
  Program sample_indirect_gt_prog = m_context->createProgramFromPTXFile(
      ptx_path, "sample_indirect_gt");
  Program ind_filt_first_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_filter_first_pass");
  Program ind_filt_second_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_filter_second_pass");
  Program display_prog = m_context->createProgramFromPTXFile(
      ptx_path, "display");
  Program display_heatmap_prog = m_context->createProgramFromPTXFile(
      ptx_path, "display_heatmaps");
  Program ind_prefilt_first_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_prefilter_first_pass");
  Program ind_prefilt_second_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_prefilter_second_pass");
  m_context->setRayGenerationProgram( 0, direct_z_sample_prog );
  m_context->setRayGenerationProgram( 1, z_filt_first_prog );
  m_context->setRayGenerationProgram( 2, z_filt_second_prog );
#ifdef GROUNDTRUTH
  m_context->setRayGenerationProgram( 3, sample_indirect_gt_prog );
#else
  m_context->setRayGenerationProgram( 3, sample_indirect_prog );
#endif
  m_context->setRayGenerationProgram( 4, ind_filt_first_prog );
  m_context->setRayGenerationProgram( 5, ind_filt_second_prog );
  m_context->setRayGenerationProgram( 6, display_prog );
  m_context->setRayGenerationProgram( 7, display_heatmap_prog );
  m_context->setRayGenerationProgram( 8, ind_prefilt_first_prog );
  m_context->setRayGenerationProgram( 9, ind_prefilt_second_prog );

  m_context["direct_ray_type"]->setUint(3u);
  m_context["indirect_ray_type"]->setUint(4u);


  //AAF GI Buffers
  uint debug_buf_type = RT_BUFFER_GPU_LOCAL;
#ifdef DEBUG_BUF
  debug_buf_type = 0;
#endif
  //Direct Illumination Buffer
  m_context["direct_illum"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //Indirect Illumination Buffer (Unaveraged RGB, and number of samples is
  // stored in A)
  m_context["indirect_illum"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height*num_buckets));
  m_context["indirect_illum_spec"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height*num_buckets));

  //Image-space Kd, Ks buffers
  m_context["Kd_image"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));
  m_context["Ks_image"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));
  m_context["phong_exp_image"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT, m_width, m_height));

  //Target SPP
  m_context["target_indirect_spp"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));

  //Z Distances to nearest contribution of indirect light
  m_context["z_dist"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT2, m_width, m_height*num_buckets));

  //Image-aligned world-space locations (used for filter weights)
  m_context["world_loc"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //Image-aligned normal vectors
  m_context["n"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //Pixels with first intersection
  m_context["visible"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_BYTE, m_width, m_height));

  //Depth buffer
  m_context["depth"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT, m_width, m_height));

  //Intermediate buffers to keep between passes
  m_context["z_dist_filter1d"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT2, m_width, m_height*num_buckets));
  m_context["indirect_illum_filter1d"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height*num_buckets));
  m_context["indirect_illum_spec_filter1d"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height*num_buckets));

  //spp buffer
  m_context["target_spb"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));
  m_context["target_spb_theoretical"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));

  m_context["target_spb_spec"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));
  m_context["target_spb_spec_theoretical"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));

  //prefiltering buffers
  m_context["prefilter_rejected"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_INT2, m_width, m_height*num_buckets));
  m_context["prefilter_rejected_filter1d"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_INT2, m_width, m_height*num_buckets));

  //specular omegavmax buffer
  m_context["spec_wvmax"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | debug_buf_type,
        RT_FORMAT_FLOAT, m_width, m_height));


  //View Mode for displaying different buffers
  m_view_mode = 0;
  m_context["view_mode"]->setUint(m_view_mode);
  m_context["view_bucket"]->setUint(0);

  //SPP Settings
  m_first_pass_spb_sqrt = 2;
  m_brute_spb = 1000;
  m_max_spb_pass = MAX_SPP/num_buckets;
  m_context["first_pass_spb_sqrt"]->setUint(m_first_pass_spb_sqrt);
  m_context["brute_spb"]->setUint(m_brute_spb);
  m_context["max_spb_pass"]->setUint(m_max_spb_pass);
  m_context["num_buckets"]->setUint(num_buckets);
  m_context["z_filter_radius"]->setInt(2);
  m_context["indirect_ray_depth"]->setUint(INDIRECT_BOUNCES);
  m_context["pixel_radius"]->setInt(10);

  //toggle settings
  m_filter_indirect = true;
  m_prefilter_indirect = true;
  m_filter_z = false;

  // Create scene geometry
  // Declare these so validation will pass
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_max_heatmap_val = 0;
  m_context["max_heatmap"]->setFloat(0);
  
#if SCENE == 0
  createSceneCornell(camera_data);
#elif SCENE == 1
  createSceneConference(camera_data);
#elif SCENE == 2
  createSceneSponza(camera_data);
#elif SCENE == 3
  createSceneCornell2(camera_data);
#elif SCENE == 4
  createSceneCornell3(camera_data);
#endif
  

  m_context["use_textures"]->setUint(m_use_textures);

#ifdef GROUNDTRUTH
  m_context["total_gt_samples_sqrt"]->setUint(GT_SAMPLES_SQRT);
  m_context["gt_samples_per_pass"]->setUint(MAX_SPP);
  m_context["gt_pass"]->setUint(0);
  int num_passes = (uint)ceil((float)GT_SAMPLES_SQRT*GT_SAMPLES_SQRT/MAX_SPP);
  m_context["gt_total_pass"]->setUint(num_passes);
  m_gt_total_pass = num_passes;
  std::cout << "GT with " << num_passes << " passes" << std::endl;
#endif

  // Finalize
  m_context->validate();
  m_context->compile();
}

bool GIScene::keyPressed( unsigned char key, int x, int y )
{
  int idelta = 1;
  int num_view_modes = 8;
#ifdef DEBUG_BUF
  num_view_modes = 16;
#endif
  switch(key)
  {

    case 'B':
    case 'b':
      m_filter_indirect = 1-m_filter_indirect;
      if (m_filter_indirect)
        std::cout << "Filter Indirect: On" << std::endl;
      else
        std::cout << "Filter Indirect: Off" << std::endl;
      return true;
    case 'M':
    case 'm':
      m_prefilter_indirect = 1-m_prefilter_indirect;
      if (m_prefilter_indirect)
        std::cout << "Prefilter Indirect: On" << std::endl;
      else
        std::cout << "Prefilter Indirect: Off" << std::endl;
      return true;
    case 'N':
    case 'n':
      m_filter_z = 1-m_filter_z;
      if (m_filter_z)
        std::cout << "Filter z: On" << std::endl;
      else
        std::cout << "Filter z: Off" << std::endl;
      return true;
    case 'S':
    case 's':
      {
        std::stringstream fname;
        fname << "output_";
        fname << std::setw(7) << std::setfill('0') << output_num;
        fname << ".ppm";
        Buffer output_buf = scene.getOutputBuffer();
        sutilDisplayFilePPM(fname.str().c_str(), output_buf->get());
        output_num++;
        std::cout << "Saved file" << fname.str() << std::endl;
        return true;
      }
    case 'D':
    case 'd':
      {
        std::stringstream fname;
        fname << "output_";
        fname << std::setw(7) << std::setfill('0') << output_num;
        fname << ".pfm";
        output_num++;

        Buffer output_buf = scene.getOutputBuffer();
        RTsize buffer_width, buffer_height;
        output_buf->getSize( buffer_width, buffer_height );



        //assuming float4, ignore alpha channel
        std::vector<float> output(buffer_width*buffer_height*4);
        unsigned char* outbuf_byte_vals = 
          reinterpret_cast<unsigned char*>(output_buf->map());
        float4* outbuf_float_vals = reinterpret_cast<float4*>(outbuf_byte_vals);
        float max_val = 0.f;

        for(int i = 0; i < buffer_width; ++i)
          for(int j = 0; j < buffer_height; ++j)
          {
            //copy every 3, ignore alpha channel
            //image is upside down
            memcpy(&output[(i+(buffer_height-j)*buffer_width)*3], 
                &outbuf_float_vals[i+j*buffer_width], sizeof(float)*3);
            max_val = max(max_val, outbuf_float_vals[i+j*buffer_width].x);
            max_val = max(max_val, outbuf_float_vals[i+j*buffer_width].y);
            max_val = max(max_val, outbuf_float_vals[i+j*buffer_width].z);
          }
        output_buf->unmap();
        
        std::ofstream ofs(fname.str().c_str(), std::ios::binary);
        ofs << "PF\n" << buffer_width << " " << buffer_height << "\n"
          << "-1.0\n";
          //<< -max_val << "\n";

        ofs.write(
            reinterpret_cast<const char*>(output.data()), 
            buffer_width*buffer_height*sizeof(float)*3);

        std::cout << "Saved file: " << fname.str() << std::endl;
        return true;
      }
    case 'Z':
      idelta = num_view_modes-1;
    case 'z':
      m_view_mode = (m_view_mode+idelta)%num_view_modes;
      m_context["view_mode"]->setUint(m_view_mode);
      switch(m_view_mode)
      {
        case 0:
          std::cout << "View mode: Normal" << std::endl;
          break;
        case 1:
          std::cout << "View mode: Direct Only" << std::endl;
          break;
        case 2:
          std::cout << "View mode: Indirect Only" << std::endl;
          break;
        case 3:
          std::cout << "View mode: Indirect Only (No Kd, Ks multiplication)" 
            << std::endl;
          break;
        case 4:
          std::cout << "View mode: Diffuse Indirect Only" << std::endl;
          break;
        case 5:
          std::cout << "View mode: Diffuse Indirect Only "
            << "(No Kd multiplication)" << std::endl;
          break;
        case 6:
          std::cout << "View mode: Specular Indirect Only" << std::endl;
          break;
        case 7:
          std::cout << "View mode: Specular Indirect Only " 
            << "(No Ks multiplication)" << std::endl;
          break;

        case 8:
          std::cout << "View mode: Z min" << std::endl;
          break;
        case 9:
          std::cout << "View mode: Z max" << std::endl;
          break;
        case 10:
          std::cout << "View mode: Target SPP/SPB" << std::endl;
          break;
        case 11:
          std::cout << "View mode: Target SPP/SPB (theoretical/unclamped)" 
            << std::endl;
          break;
        case 12:
          std::cout << "View mode: Rejected Sample Ratio" 
            << std::endl;
          break;
        case 13:
          std::cout << "View mode: Specular omega v max" 
            << std::endl;
          break;
        case 14:
          std::cout << "View mode: Target Specular SPP/SPB" << std::endl;
          break;
        case 15:
          std::cout << "View mode: Target Specular SPP/SPB " <<
            "(theoretical/unclamped)" << std::endl;
          break;
      }
      return true;
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      m_context["view_bucket"]->setUint(key-'0');
      std::cout << "Viewing bucket: " << key << std::endl;
      return true;
    case 'v':
    case 'V':
#ifndef DEBUG_BUF
      std::cout << "SPP Values: Please turn on debug buffers" << std::endl;
#endif
      {
        Buffer buffer = m_context["output_buffer"]->getBuffer();
        RTsize buffer_width, buffer_height;
        RTsize bucket_buffer_height = buffer_height * num_buckets;
        Buffer spb_buf = m_context["target_spb"]->getBuffer();
        Buffer spb_theo_buf = m_context["target_spb_theoretical"]->getBuffer();
        float* spb_vals = reinterpret_cast<float*>(spb_buf->map());
        float* spb_theo_vals = reinterpret_cast<float*>(spb_theo_buf->map());
        float max_spb = spb_theo_vals[0];
        float min_spb = spb_theo_vals[0];
        float max_clamped_spb = spb_vals[0];
        float min_clamped_spb = spb_vals[0];
        float average_spb = 0;
        float average_clamped_spb = 0;
        for(int i = 0; i < buffer_width; ++i)
          for(int j = 0; j < bucket_buffer_height; ++j)
          {
            float cur_spb_theo_val = max(spb_theo_vals[i+j*buffer_width],0);
            float cur_spb_val = max(spb_vals[i+j*buffer_width],0);

            min_spb = min(min_spb, cur_spb_theo_val);
            max_spb = max(max_spb, cur_spb_theo_val);
            average_spb += cur_spb_val;

            min_clamped_spb = min(min_clamped_spb, cur_spb_val);
            max_clamped_spb = max(max_clamped_spb, cur_spb_val);
            average_clamped_spb += cur_spb_val;
          }
        average_spb /= buffer_width*bucket_buffer_height;
        average_clamped_spb /= buffer_width*bucket_buffer_height;

        float max_spp = 0;
        float min_spp = max_spb*num_buckets;
        float max_clamped_spp = 0;
        float min_clamped_spp = max_clamped_spb*num_buckets;
        float average_spp = 0;
        float average_clamped_spp = 0;
        for(int i = 0; i < buffer_width; ++i)
          for(int j = 0; j < buffer_height; ++j)
          {
            float cur_spp = 0;
            float cur_clamped_spp = 0;
            for(int k = 0; k < num_buckets; ++k)
            {
              float cur_spb_theo_val = max(
                  spb_vals[i+(j*num_buckets+k)*buffer_width],0);
              float cur_spb_val = max(
                  spb_vals[i+(j*num_buckets+k)*buffer_width],0);

              cur_spp += cur_spb_theo_val;
              cur_clamped_spp += cur_spb_val;
            }
            min_spp = min(min_spp, cur_spp);
            max_spp = max(max_spp, cur_spp);
            max_clamped_spp = max(max_clamped_spp, cur_clamped_spp);
            min_clamped_spp = min(min_clamped_spp, cur_clamped_spp);
            average_spp += cur_spp;
            average_clamped_spp += cur_clamped_spp;
          }
        average_spp /= buffer_width * buffer_height;
        average_clamped_spp /= buffer_width * buffer_height;

        spb_buf->unmap();
        spb_theo_buf->unmap();

        std::cout << "Buckets:" << std::endl;
        std::cout << "  Average SPB: " << average_clamped_spb << std::endl;
        std::cout << "  Maximum SPB: " << max_clamped_spb << std::endl;
        std::cout << "  Minimum SPB: " << min_clamped_spb << std::endl;
        std::cout << "  Average SPB (Theoretical): " << 
          average_spb << std::endl;
        std::cout << "  Maximum SPB (Theoretical): " << max_spb << std::endl;
        std::cout << "  Minimum SPB (Theoretical): " << min_spb << std::endl;
        std::cout << std::endl;

        std::cout << "Pixels:" << std::endl;
        std::cout << "  Average SPP: " << average_clamped_spp << std::endl;
        std::cout << "  Maximum SPP: " << max_clamped_spp << std::endl;
        std::cout << "  Minimum SPP: " << min_clamped_spp << std::endl;
        std::cout << "  Average SPP (Theoretical): " << average_spp 
          << std::endl; 
        std::cout << "  Maximum SPP (Theoretical): " << max_spp << std::endl;
        std::cout << "  Minimum SPP (Theoretical): " << min_spp << std::endl;

      }
      return true;
    case 'X':
    case 'x':
#ifndef DEBUG_BUF
      std::cout << "Max heatmap value: Please turn on debug buffers" 
        << std::endl;
#endif
      std::cout << "Max heatmap value: " << m_max_heatmap_val << std::endl;
      return true;

  }
  return false;
}

#ifndef GROUNDTRUTH
void GIScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  if( m_camera_changed ) {
    m_camera_changed = false;
    m_frame = 1;
  }

  m_context["frame_number"]->setUint( m_frame++ );

  RTsize bucket_buffer_height = buffer_height * num_buckets;
  m_context->launch( 0, static_cast<unsigned int>(buffer_width), 
      static_cast<unsigned int>(buffer_height));
  if (m_filter_z)
  {
    m_context->launch( 1, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(bucket_buffer_height));
    m_context->launch( 2, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(bucket_buffer_height));
  }
  if (m_prefilter_indirect)
  {
    m_context->launch( 8, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(bucket_buffer_height));
    m_context->launch( 9, static_cast<unsigned int>(buffer_width),
        static_cast<unsigned int>(bucket_buffer_height));
  }
  m_context->launch( 3, static_cast<unsigned int>(buffer_width),
      static_cast<unsigned int>(bucket_buffer_height));
  if (m_filter_indirect)
  {
    m_context->launch( 4, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(bucket_buffer_height));
    m_context->launch( 5, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(bucket_buffer_height));
  }
  m_context->launch( 6, static_cast<unsigned int>(buffer_width), 
      static_cast<unsigned int>(buffer_height));
#ifdef DEBUG_BUF
  if (m_view_mode > 7)
  {
    //scale heatmaps
    m_max_heatmap_val = 0;
    if (m_view_mode == 8 || m_view_mode == 9)
    {
      Buffer z_distbuf = m_context["z_dist"]->getBuffer();
      float2* z_dists = reinterpret_cast<float2*>(z_distbuf->map());
      for(int i = 0; i < buffer_width; ++i)
        for(int j = 0; j < bucket_buffer_height; ++j)
          if (m_view_mode == 4)
            m_max_heatmap_val = max(m_max_heatmap_val, 
                z_dists[i+j*buffer_width].x);
          else
            m_max_heatmap_val = max(m_max_heatmap_val, 
                z_dists[i+j*buffer_width].y);
      z_distbuf->unmap();
    }
    if (m_view_mode == 10)
    {
      Buffer spb_buf = m_context["target_spb"]->getBuffer();
      float* spb_vals = reinterpret_cast<float*>(spb_buf->map());
      for(int i = 0; i < buffer_width; ++i)
        for(int j = 0; j < bucket_buffer_height; ++j)
        {
          //reject values that are too large
          float cur_spb_val = spb_vals[i+j*buffer_width];
          if (cur_spb_val < 10000.)
            m_max_heatmap_val = max(m_max_heatmap_val, cur_spb_val);
        }
      spb_buf->unmap();
    }
    if (m_view_mode == 11)
    {
      Buffer spb_buf = m_context["target_spb_theoretical"]->getBuffer();
      float* spb_vals = reinterpret_cast<float*>(spb_buf->map());
      for(int i = 0; i < buffer_width; ++i)
        for(int j = 0; j < bucket_buffer_height; ++j)
        {
          //reject values that are too large
          float cur_spb_val = spb_vals[i+j*buffer_width];
          if (cur_spb_val < 10000.)
            m_max_heatmap_val = max(m_max_heatmap_val, cur_spb_val);
        }
      spb_buf->unmap();
    }
    if (m_view_mode == 12)
    {
      m_max_heatmap_val = 0.25f;
    }
    if (m_view_mode == 13)
    {
      Buffer spec_wvmax_buf = m_context["spec_wvmax"]->getBuffer();
      float* spec_wvmax_vals = reinterpret_cast<float*>(spec_wvmax_buf->map());
      for (int i = 0; i < buffer_width; ++i)
        for (int j = 0; j < buffer_height; ++j)
        {
          m_max_heatmap_val = max(m_max_heatmap_val, 
              min(spec_wvmax_vals[i+j*buffer_width],1000.));
        }
      spec_wvmax_buf->unmap();
    }
    if (m_view_mode == 14)
    {
      Buffer spb_buf = m_context["target_spb_spec"]->getBuffer();
      float* spb_vals = reinterpret_cast<float*>(spb_buf->map());
      for(int i = 0; i < buffer_width; ++i)
        for(int j = 0; j < bucket_buffer_height; ++j)
        {
          //reject values that are too large
          float cur_spb_val = spb_vals[i+j*buffer_width];
          if (cur_spb_val < 10000.)
            m_max_heatmap_val = max(m_max_heatmap_val, cur_spb_val);
        }
      spb_buf->unmap();
    }
    if (m_view_mode == 15)
    {
      Buffer spb_buf = m_context["target_spb_spec_theoretical"]->getBuffer();
      float* spb_vals = reinterpret_cast<float*>(spb_buf->map());
      for(int i = 0; i < buffer_width; ++i)
        for(int j = 0; j < bucket_buffer_height; ++j)
        {
          //reject values that are too large
          float cur_spb_val = spb_vals[i+j*buffer_width];
          if (cur_spb_val < 10000.)
            m_max_heatmap_val = max(m_max_heatmap_val, cur_spb_val);
        }
      spb_buf->unmap();
    }
    m_context["max_heatmap"]->setFloat(m_max_heatmap_val);
    
    m_context->launch( 7, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(buffer_height));
  }
#endif

}
#else
void GIScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  if( m_camera_changed ) {
    m_camera_changed = false;
    m_frame = 1;
  }

  m_context["frame_number"]->setUint( m_frame++ );

  RTsize bucket_buffer_height = buffer_height * num_buckets;
  if (m_frame == 2)
  {
    m_context->launch( 0, static_cast<unsigned int>(buffer_width), 
        static_cast<unsigned int>(buffer_height));
    for (int i = 0; i < m_gt_total_pass; ++i)
    {
      m_context["gt_pass"]->setUint(i);
      m_context->launch( 3, static_cast<unsigned int>(buffer_width),
          static_cast<unsigned int>(buffer_height));
    }
  }
  m_context->launch( 6, static_cast<unsigned int>(buffer_width), 
      static_cast<unsigned int>(buffer_height));
}
#endif


//-----------------------------------------------------------------------------

Buffer GIScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

GeometryInstance GIScene::createParallelogram( const float3& anchor,
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

GeometryInstance GIScene::createLightParallelogram( const float3& anchor,
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

void GIScene::setMaterial( GeometryInstance& gi,
                                   Material material,
                                   const std::string& color_name,
                                   const float3& color)
{
  gi->addMaterial(material);
  gi[color_name]->setFloat(color);
}

void GIScene::createSceneCornell(InitialCameraData& camera_data)
{


  m_use_textures = false;
  // Set up camera
  const float vfov = 35.f;

  //cornell
  camera_data = InitialCameraData( 
      make_float3( 278.0f, 273.0f, -800.0f ), // eye
      make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      vfov );                                // vfov

  // Light buffer
  ParallelogramLight light;
  light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
  light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
  light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
  light.normal   = normalize( cross(light.v1, light.v2) );
  light.emission = make_float3( 15.0f, 15.0f, 5.0f );

  Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( ParallelogramLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer );
  // Set up material
  Material diffuse = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );
  
  //dummy texture maps
  diffuse["ambient_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.2f, 0.2f, 0.2f ) ) );
  diffuse["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.8f, 0.8f, 0.8f ) ) );
  diffuse["specular_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.0f, 0.0f, 0.0f ) ) );
  diffuse["Kd"]->setFloat(0.5,0.5,1.5);
  diffuse["Ks"]->setFloat(0.,0.,0.);
  diffuse["phong_exp"]->setFloat(1.f);

  // Set up parallelogram programs
  std::string ptx_path = ptxpath( "aaf_gi", "parallelogram.cu" );
  m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, 
      "bounds" );
  m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, 
      "intersect" );

  // create geometry instances
  std::vector<GeometryInstance> gis;

  const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
  const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
  const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
  const float3 black = make_float3( 0.f, 0.f, 0.f );
  const float3 light_em = make_float3( 15.0f, 15.0f, 5.0f );

  // Floor
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Ceiling
  gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Back wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
                                      make_float3( 0.0f, 548.8f, 0.0f),
                                      make_float3( 556.0f, 0.0f, 0.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Right wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", green);

  // Left wall
  gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", red);

  // Short block
  gis.push_back( createParallelogram( make_float3( 130.0f, 165.0f, 65.0f),
                                      make_float3( -48.0f, 0.0f, 160.0f),
                                      make_float3( 160.0f, 0.0f, 49.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 290.0f, 0.0f, 114.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( -50.0f, 0.0f, 158.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 130.0f, 0.0f, 65.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( 160.0f, 0.0f, 49.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 82.0f, 0.0f, 225.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( 48.0f, 0.0f, -160.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 240.0f, 0.0f, 272.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( -158.0f, 0.0f, -47.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Tall block
  gis.push_back( createParallelogram( make_float3( 423.0f, 330.0f, 247.0f),
                                      make_float3( -158.0f, 0.0f, 49.0f),
                                      make_float3( 49.0f, 0.0f, 159.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 423.0f, 0.0f, 247.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( 49.0f, 0.0f, 159.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 472.0f, 0.0f, 406.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( -158.0f, 0.0f, 50.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 314.0f, 0.0f, 456.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( -49.0f, 0.0f, -160.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);
  gis.push_back( createParallelogram( make_float3( 265.0f, 0.0f, 296.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( 158.0f, 0.0f, -49.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Create shadow group (no light)
  GeometryGroup shadow_group = m_context->createGeometryGroup(gis.begin(), 
      gis.end());
  shadow_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
  m_context["top_shadower"]->set( shadow_group );

  // Create geometry group
  GeometryGroup geometry_group = m_context->createGeometryGroup(gis.begin(), 
      gis.end());
  geometry_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh"));
  m_context["top_object"]->set( geometry_group );
  m_context["vfov"]->setFloat( vfov );
}


void GIScene::createSceneCornell2(InitialCameraData& camera_data)
{

  m_use_textures = false;
  // Set up camera
  const float vfov = 35.f;

  camera_data = InitialCameraData( 
      make_float3( 278.0f, 550.0f, -700.0f ), // eye
      make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      35.0f );         // vfov



  // Light buffer
  ParallelogramLight light;
  light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
  light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
  light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
  light.normal   = normalize( cross(light.v1, light.v2) );
  light.emission = make_float3( 15.0f, 15.0f, 5.0f );

  Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( ParallelogramLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer );
  // Set up material
  Material diffuse = m_context->createMaterial();
  Material diffuse2 = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );
  diffuse2->setClosestHitProgram( 3, diffuse_p );
  diffuse2->setAnyHitProgram( 1, diffuse_ah );
  
  //dummy texture maps
  diffuse["ambient_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.2f, 0.2f, 0.2f ) ) );
  diffuse["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.8f, 0.8f, 0.8f ) ) );
  diffuse["specular_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.0f, 0.0f, 0.0f ) ) );
  diffuse["Kd"]->setFloat(0.5,0.5,1.5);
  diffuse["Ks"]->setFloat(0.,0.,0.);
  diffuse["phong_exp"]->setFloat(1.);
  diffuse2["ambient_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.2f, 0.2f, 0.2f ) ) );
  diffuse2["diffuse_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.8f, 0.8f, 0.8f ) ) );
  diffuse2["specular_map"]->setTextureSampler( loadTexture( m_context, "", 
        make_float3( 0.0f, 0.0f, 0.0f ) ) );

  diffuse2["Kd"]->setFloat(0.0,0.0,0.0);
  diffuse2["Ks"]->setFloat(1.,1.,1.);
  diffuse2["phong_exp"]->setFloat(100.);

  // Set up parallelogram programs
  std::string ptx_path = ptxpath( "aaf_gi", "parallelogram.cu" );
  m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, 
      "bounds" );
  m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, 
      "intersect" );

  // create geometry instances
  std::vector<GeometryInstance> gis;

  const float3 white = make_float3( 1.f, 1.f, 1.f );
  const float3 green = make_float3( 0.2f, 1.f, 0.2f );
  const float3 red   = make_float3( 1.f, 0.2f, 0.2f );
  const float3 light_em = make_float3( 15.0f, 15.0f, 15.0f );

  // Floor
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Ceiling
  gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Back wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
                                      make_float3( 0.0f, 548.8f, 0.0f),
                                      make_float3( 556.0f, 0.0f, 0.0f) ) );
  setMaterial(gis.back(), diffuse, "Kd", white);

  // Right wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", green);

  // Left wall
  gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ) ) );
  setMaterial(gis.back(), diffuse, "Kd", red);

  GeometryGroup geom_group = m_context->createGeometryGroup(gis.begin(), 
      gis.end());

  Matrix4x4 dragon_xform = Matrix4x4::translate(make_float3(370,0,270)) 
    * Matrix4x4::rotate(M_PI + M_PI/3,make_float3(0,1,0))
	* Matrix4x4::scale(make_float3(28,28,28));
    
  std::string objpath = std::string( sutilSamplesDir() ) +
    "/gi_research2/data/dragon.obj";
  ObjLoader * dragon_loader = new ObjLoader( objpath.c_str(), m_context, 
      geom_group, diffuse, true );
  dragon_loader->load(dragon_xform);

  Matrix4x4 elephant_xform = Matrix4x4::translate(make_float3(150,0,270))	
    * Matrix4x4::rotate(M_PI-M_PI/8,make_float3(0,-1,0))
    * Matrix4x4::scale(make_float3(0.35,0.35,0.35));
    
  objpath = std::string( sutilSamplesDir() ) + 
    "/gi_research2/data/elephant.obj";
  ObjLoader * elephant_loader = new ObjLoader( objpath.c_str(), m_context, 
      geom_group, diffuse2, true );
  elephant_loader->load(elephant_xform);

   // Declare these so validation will pass
  m_context["vfov"]->setFloat( vfov );
  m_context["top_object"]->set( geom_group );
  m_context["top_shadower"]->set( geom_group );
}



void GIScene::createSceneSponza(InitialCameraData& camera_data)
{
  m_use_textures = true;
  // Set up camera
  const float vfov = 35.f;
  //sponza
  camera_data = InitialCameraData( make_float3( -542.f, 520.f, 162.f ), // eye
      make_float3( 166.f, 202.f, 251.f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      vfov );                                // vfov


  ParallelogramLight light;
  light.corner   = make_float3( 580.0f, 500.f, 0.f);
  light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
  light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
  light.normal   = normalize( cross(light.v1, light.v2) );
  light.emission = make_float3( 15.0f, 15.0f, 5.0f );
  
  Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( ParallelogramLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer ); 
  
  GeometryGroup conference_geom_group = m_context->createGeometryGroup();
  

  Material diffuse = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );

  diffuse["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  diffuse["Ks"]->setFloat( 0.f, 0.f, 0.f );
  diffuse["phong_exp"]->setFloat( 1.f );
  diffuse["obj_id"]->setInt(10);
  
  std::string objpath = std::string( sutilSamplesDir() ) + 
    "/gi_research2/data/crytek_sponza/sponza_noflag.obj";
  ObjLoader * conference_loader = new ObjLoader( objpath.c_str(), 
      m_context, conference_geom_group, diffuse, true );
  conference_loader->load();


  // Declare these so validation will pass
  m_context["vfov"]->setFloat( vfov );

  
  m_context["top_object"]->set( conference_geom_group );
  m_context["top_shadower"]->set( conference_geom_group );

}

void GIScene::createSceneConference(InitialCameraData& camera_data)
{
  m_use_textures = true;
  // Set up camera
  const float vfov = 35.f;
  //conference
  camera_data = InitialCameraData( make_float3( -542.f, 520.f, 162.f ), // eye
  make_float3( 166.f, 202.f, 251.f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      vfov );                                // vfov

  ParallelogramLight light;
  light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
  light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
  light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
  light.normal   = normalize( cross(light.v1, light.v2) );
  light.emission = make_float3( 15.0f, 15.0f, 5.0f );
  
  Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( ParallelogramLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer ); 
  
  GeometryGroup conference_geom_group = m_context->createGeometryGroup();
  

  Material diffuse = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );

  diffuse["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  diffuse["Ks"]->setFloat( 0.f, 0.f, 0.f );
  diffuse["phong_exp"]->setFloat( 1.f );
  
  std::string objpath = std::string( sutilSamplesDir() ) + 
    "/gi_research2/data/conference/conference_diffuse.obj";
  ObjLoader * conference_loader = new ObjLoader( objpath.c_str(), 
      m_context, conference_geom_group, diffuse, true );
  conference_loader->load();

  m_context["vfov"]->setFloat( vfov );
  
  m_context["top_object"]->set( conference_geom_group );
  m_context["top_shadower"]->set( conference_geom_group );
}

void GIScene::createSceneCornell3(InitialCameraData& camera_data)
{
  m_use_textures = true;
  // Set up camera
  const float vfov = 35.f;
  
  camera_data = InitialCameraData( 
      make_float3( 0.f, 1.0f, 3.0f ), // eye
      make_float3( 0.0f, 0.75f, 0.0f ),    // lookat
      make_float3( 0.0f, 1.0f,  0.0f ),       // up
      vfov );                                // vfov

  ParallelogramLight light;
  light.corner   = make_float3( 0.0f, 1.5f, 0.0f);
  light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
  light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
  light.normal   = normalize( cross(light.v1, light.v2) );
  light.emission = make_float3( 15.0f, 15.0f, 5.0f );
  
  Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  light_buffer->setFormat( RT_FORMAT_USER );
  light_buffer->setElementSize( sizeof( ParallelogramLight ) );
  light_buffer->setSize( 1u );
  memcpy( light_buffer->map(), &light, sizeof( light ) );
  light_buffer->unmap();
  m_context["lights"]->setBuffer( light_buffer ); 
  
  GeometryGroup conference_geom_group = m_context->createGeometryGroup();
  

  Material diffuse = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", 
        "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );

  diffuse["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  diffuse["Ks"]->setFloat( 0.f, 0.f, 0.f );
  diffuse["phong_exp"]->setFloat( 1.f );
  
  std::string objpath = std::string( sutilSamplesDir() ) + 
    "/gi_research2/data/cornell_box/CornellBox-Glossy.obj";
  ObjLoader * conference_loader = new ObjLoader( objpath.c_str(), 
      m_context, conference_geom_group, diffuse, true );
  conference_loader->load();

  m_context["vfov"]->setFloat( vfov );
  
  m_context["top_object"]->set( conference_geom_group );
  m_context["top_shadower"]->set( conference_geom_group );
}

//-----------------------------------------------------------------------------
//
// main
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               "
    << "Print this usage message\n"

    << " -rrd | --rr-begin-depth <d>                 "
    << "Start Russian Roulette killing of rays at depth <d>\n"

    << "  -md | --max-depth <d>                      "
    << "Maximum ray tree depth\n"
    << "  -n  | --sqrt_num_samples <ns>              "
    << "Number of samples to perform for each frame\n"
    << "  -t  | --timeout <sec>                      "
    << "Seconds before stopping rendering. Set to 0 for no stopping.\n"
    << std::endl;

  std::cout
    << "Axis-Aligned Filtering key bindings: " << std::endl
    << "b: Toggle indirect illumination filtering" << std::endl
    << "n: Toggle z filtering" << std::endl
    << "m: Toggle indirect illumination pre-filtering" << std::endl
    << "Z/z: Switch between debug views. " << std::endl
    << "Number Keys: Switch between buckets in supported debug views. " 
    << std::endl
    << "0: View combined bucket in supported debug views" << std::endl
    << "s: Save current buffer to file (PPM). " << std::endl
    << "d: Save current buffer to file (PFM). " << std::endl
    << "x: Print maximum heatmap value if viewing a heatmap" << std::endl;
  //GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}


unsigned int getUnsignedArg(int& arg_index, int argc, char** argv)
{
  int result = -1;
  if (arg_index+1 < argc) {
    result = atoi(argv[arg_index+1]);
  } else {
    std::cerr << "Missing argument to "<<argv[arg_index]<<"\n";
    printUsageAndExit(argv[0]);
  }
  if (result < 0) {
    std::cerr << "Argument to "<<argv[arg_index]<<" must be positive.\n";
    printUsageAndExit(argv[0]);
  }
  ++arg_index;
  return static_cast<unsigned int>(result);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  unsigned int sqrt_num_samples = 2u;

  unsigned int width = WIDTH, height = HEIGHT;
  unsigned int rr_begin_depth = 2u;
  unsigned int max_depth = 100u;
  float timeout = 10.0f;

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--rr-begin-depth" || arg == "-rrd" ) {
      rr_begin_depth = getUnsignedArg(i, argc, argv);
    } else if ( arg == "--max-depth" || arg == "-md" ) {
      max_depth = getUnsignedArg(i, argc, argv);
    } else if ( arg == "--sqrt_num_samples" || arg == "-n" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      sqrt_num_samples = atoi( argv[++i] );
    } else if ( arg == "--timeout" || arg == "-t" ) {
      if(++i < argc) {
        timeout = static_cast<float>(atof(argv[i]));
      } else {
        std::cerr << "Missing argument to "<<arg<<"\n";
        printUsageAndExit(argv[0]);
      }
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    scene.setNumSamples( sqrt_num_samples );
    scene.setDimensions( width, height );
    GLUTDisplay::setProgressiveDrawingTimeout(0);
    //GLUTDisplay::setUseSRGB(true);
    GLUTDisplay::run( "Axis-Aligned Filtering: Global Illumination", 
        &scene, GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
