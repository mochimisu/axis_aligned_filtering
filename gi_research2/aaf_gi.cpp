
/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

//------------------------------------------------------------------------------
//
// aaf_gi.cpp: render cornell box using path tracing.
//
//------------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <GLUTDisplay.h>
#include <PPMLoader.h>
#include <sampleConfig.h>
#include <sstream>
#include <iostream>
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
    return std::string(sutilSamplesPtxDir()) + "/aaf_gi_generated_" + base + ".ptx";
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

  void   setNumSamples( unsigned int sns )                           { m_sqrt_num_samples= sns; }
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }

private:
  // Should return true if key was handled, false otherwise.
  virtual bool keyPressed(unsigned char key, int x, int y);
  void createGeometry();

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

  bool m_filter_indirect;
  bool m_filter_z;
};

GIScene scene;
int output_num = 0;
const unsigned int num_buckets = 4;

void GIScene::initScene( InitialCameraData& camera_data )
{
  m_context->setRayTypeCount( 5 );
  m_context->setEntryPointCount( 7 );
  m_context->setStackSize( 1800 );

  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["max_depth"]->setUint(m_max_depth);
  m_context["pathtrace_ray_type"]->setUint(0u);
  m_context["pathtrace_shadow_ray_type"]->setUint(1u);
  m_context["pathtrace_bsdf_shadow_ray_type"]->setUint(2u);
  m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);


  // Setup output buffer
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, m_width, m_height );
  output_buffer->set(buffer);


  // Set up camera
  const float vfov = 35.f;
  /*
  //sponza
  camera_data = InitialCameraData( make_float3( 652.5f, 693.5f, 0.f ), // eye
  make_float3( 614.0f, 654.0f, 0.0f ),    // lookat
  make_float3( 0.0f, 1.0f,  0.0f ),       // up
  35.0f );                                // vfov
  */

  camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -800.0f ), // eye
                                   make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
                                   make_float3( 0.0f, 1.0f,  0.0f ),       // up
                                   vfov );                                // vfov

  // Declare these so validation will pass
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["vfov"]->setFloat( vfov );

  m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3(0.0f) );

  // Setup programs
  std::string ptx_path = ptxpath( "aaf_gi", "aaf_gi.cu" );
  //Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pathtrace_camera" );
  //m_context->setRayGenerationProgram( 0, ray_gen_program );
  Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );

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
  Program ind_filt_first_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_filter_first_pass");
  Program ind_filt_second_prog = m_context->createProgramFromPTXFile(
      ptx_path, "indirect_filter_second_pass");
  Program display_prog = m_context->createProgramFromPTXFile(
      ptx_path, "display");
  m_context->setRayGenerationProgram( 0, direct_z_sample_prog );
  m_context->setRayGenerationProgram( 1, z_filt_first_prog );
  m_context->setRayGenerationProgram( 2, z_filt_second_prog );
  m_context->setRayGenerationProgram( 3, sample_indirect_prog );
  m_context->setRayGenerationProgram( 4, ind_filt_first_prog );
  m_context->setRayGenerationProgram( 5, ind_filt_second_prog );
  m_context->setRayGenerationProgram( 6, display_prog );

  m_context["direct_ray_type"]->setUint(3u);
  m_context["indirect_ray_type"]->setUint(4u);


  //AAF GI Buffers
  //Direct Illumination Buffer
  m_context["direct_illum"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //Indirect Illumination Buffer (Unaveraged RGB, and number of samples is
  // stored in A)
  m_context["indirect_illum"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height*num_buckets));

  //Image-space Kd, Ks buffers
  m_context["Kd_image"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));
  m_context["Ks_image"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //Target SPP
  m_context["target_indirect_spp"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT, m_width, m_height*num_buckets));

  //Z Distances to nearest contribution of indirect light
  m_context["z_dist"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
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

  //debug buffer
  m_context["debug_buf"]->set(
      m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
        RT_FORMAT_FLOAT3, m_width, m_height));

  //random numbers

  Buffer indirect_rng_seeds = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT2, m_width, m_height*num_buckets);
  m_context["indirect_rng_seeds"]->set(indirect_rng_seeds);
  uint2* seeds = reinterpret_cast<uint2*>( indirect_rng_seeds->map() );
  for(unsigned int i = 0; i < m_width * m_height*num_buckets; ++i )
  seeds[i] = random2u();
  indirect_rng_seeds->unmap();

  //View Mode for displaying different buffers
  m_view_mode = 0;
  m_context["view_mode"]->setUint(m_view_mode);

  //SPP Settings
  m_first_pass_spb_sqrt = 2;
  m_brute_spb = 1000;
  m_max_spb_pass = 25;
  m_context["first_pass_spb_sqrt"]->setUint(m_first_pass_spb_sqrt);
  m_context["brute_spb"]->setUint(m_brute_spb);
  m_context["max_spb_pass"]->setUint(m_max_spb_pass);
  m_context["num_buckets"]->setUint(num_buckets);
  m_context["z_filter_radius"]->setInt(2);
  m_context["indirect_ray_depth"]->setUint(1);
  m_context["pixel_radius"]->setInt(10);

  //toggle settings
  m_filter_indirect = true;
  m_filter_z = true;

  

  // Create scene geometry
  createGeometry();

  // Finalize
  m_context->validate();
  m_context->compile();
}

bool GIScene::keyPressed( unsigned char key, int x, int y )
{
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
			std::cout << "Saved file" << std::endl;
			return true;
		}	
	}
  return false;
}

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

  /*
  for (int pass = 0; pass < 7; ++pass)
  {
    m_context->launch( pass,
        static_cast<unsigned int>(buffer_width),
        static_cast<unsigned int>(buffer_height)
        );
  }
  */
  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height));
  if (m_filter_z)
  {
	  m_context->launch( 1, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height*num_buckets));
	  m_context->launch( 2, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height*num_buckets));
  }
  m_context->launch( 3, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height*num_buckets));
  if (m_filter_indirect)
  {
	  m_context->launch( 4, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height*num_buckets));
	  m_context->launch( 5, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height*num_buckets));
  }
  m_context->launch( 6, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height));
}

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

void GIScene::createGeometry()
{
	
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
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );

  // Set up parallelogram programs
  std::string ptx_path = ptxpath( "aaf_gi", "parallelogram.cu" );
  m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
  m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

  // create geometry instances
  std::vector<GeometryInstance> gis;

  const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
  const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
  const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
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
  GeometryGroup shadow_group = m_context->createGeometryGroup(gis.begin(), gis.end());
  shadow_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
  m_context["top_shadower"]->set( shadow_group );

  // Create geometry group
  GeometryGroup geometry_group = m_context->createGeometryGroup(gis.begin(), gis.end());
  geometry_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
  m_context["top_object"]->set( geometry_group );

  /*
  
  //sponza
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
  
  Group _top_grp = m_context->createGroup();
  GeometryGroup sponza_geom_group = m_context->createGeometryGroup();
  

  Material diffuse = m_context->createMaterial();
  Program diffuse_ah = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", "aaf_gi.cu" ), "shadow" );
  Program diffuse_p = m_context->createProgramFromPTXFile( ptxpath( "aaf_gi", "aaf_gi.cu" ), "closest_hit_direct" );
  diffuse->setClosestHitProgram( 3, diffuse_p );
  diffuse->setAnyHitProgram( 1, diffuse_ah );

  diffuse["Kd"]->setFloat( 0.87402f, 0.87402f, 0.87402f );
  diffuse["obj_id"]->setInt(10);
  
  //ObjLoader * floor_loader = new ObjLoader( texpath("dabrovic-sponza/sponza.obj").c_str(), m_context, sponza_geom_group, floor_mat );
  std::string objpath = std::string( sutilSamplesDir() ) + "/gi_research/data/crytek_sponza/sponza.obj";
  ObjLoader * floor_loader = new ObjLoader( objpath.c_str(), m_context, sponza_geom_group, diffuse, true );
  floor_loader->load();
  
  
  _top_grp->setChildCount(1);
  _top_grp->setChild(0, sponza_geom_group);
  
  _top_grp->setAcceleration(m_context->createAcceleration("Bvh", "Bvh") );
  _top_grp->getAcceleration()->setProperty("refit", "1");
  
  
  m_context["top_object"]->set( _top_grp );
  m_context["top_shadower"]->set( _top_grp );
  */
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
    << "  -h  | --help                               Print this usage message\n"
    << " -rrd | --rr-begin-depth <d>                 Start Russian Roulette killing of rays at depth <d>\n"
    << "  -md | --max-depth <d>                      Maximum ray tree depth\n"
    << "  -n  | --sqrt_num_samples <ns>              Number of samples to perform for each frame\n"
    << "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
    << std::endl;
  GLUTDisplay::printUsage();

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

  unsigned int width = 512u, height = 512u;
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
    GLUTDisplay::setProgressiveDrawingTimeout(timeout);
    GLUTDisplay::setUseSRGB(true);
    GLUTDisplay::run( "Cornell Box Scene", &scene, GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
