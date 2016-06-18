#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/torchzfp.cpp"
#else

#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )

#undef TAPI
#define TAPI __declspec(dllimport)

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#include <stdint.h>
#include <memory>
extern "C" {
  #include "zfp.h"
}

static int torchzfp_(Main_compress)(lua_State *L) {
  THTensor* in = reinterpret_cast<THTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  real* in_data = THTensor_(data)(in);
  const uint32_t dim = in->nDimension;
  if (dim == 0) {
    luaL_error(L, "Input tensor must not be empty");
  }
  THByteTensor* out = reinterpret_cast<THByteTensor*>(
      luaT_checkudata(L, 2, "torch.ByteTensor"));
  const double accuracy = static_cast<double>(lua_tonumber(L, 3));

  // Hacky code to figure out what type 'real' is at runtime. This really should
  // be template specialization (so it's compiled in at runtime).
  real dummy;
  static_cast<void>(dummy);  // Silence compiler warnings.
  zfp_type type;
  if (typeid(dummy) == typeid(float)) {
    type = zfp_type_float;
  } else if (typeid(dummy) == typeid(double)) {
    type = zfp_type_double;
  } else {
    luaL_error(L, "Input type must be double or float.");
  }

  // Allocate meta data for the array.
  zfp_field* field;
  uint32_t dim_zfp;
  if (dim == 1) {
    field = zfp_field_1d(in_data, type, in->size[0]);
    dim_zfp = 1;
  } else if (dim == 2) {
    field = zfp_field_2d(in_data, type, in->size[1], in->size[0]);
    dim_zfp = 2;
  } else if (dim == 3) {
    field = zfp_field_3d(in_data, type, in->size[2], in->size[1], in->size[0]);
    dim_zfp = 3;
  } else {
    // ZFP only allows up to 3D tensors, so we'll have to treat the input
    // tensor as a concatenated 3D tensor. This will affect compression ratios
    // but there's not much we can do about this.
    uint32_t sizez = 1;
    for (uint32_t i = 0; i < dim - 2; i++) {
      sizez *= in->size[i];
    }
    uint32_t sizey = in->size[dim - 2];
    uint32_t sizex = in->size[dim - 1];
    field = zfp_field_3d(in_data, type, sizex, sizey, sizez);
    dim_zfp = 4;
  }

  // Allocate meta data for the compressed stream.
  zfp_stream* zfp = zfp_stream_open(NULL);

  // Set stream compression mode and parameters.
  zfp_stream_set_accuracy(zfp, accuracy, type);

  // Allocate buffer for compressed data.
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[bufsize]);

  // Associate bit stream with allocated buffer.
  bitstream* stream = stream_open(buffer.get(), bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  // Compress entire array.
  const size_t zfpsize = zfp_compress(zfp, field);

  // Clean up.
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);

  if (!zfpsize) {
    luaL_error(L, "ZFP compression failed!");
  }

  // Copy the compressed array into the return tensor. NOTE: Torch does not
  // support in-place resize with shrink. If you resize smaller you ALWAYS
  // keep around the memory, so unfortuantely this copy is necessary (i.e.
  // we will always need to perform the compression in a temporary buffer
  // first).
  THByteTensor_resize1d(out, zfpsize);
  unsigned char* out_data = THByteTensor_data(out);
  memcpy(out_data, buffer.get(), zfpsize);

  return 0;  // Recall: number of lua return items.
}

static int torchzfp_(Main_decompress)(lua_State *L) {
  THTensor* out = reinterpret_cast<THTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  real* out_data = THTensor_(data)(out);
  const uint32_t dim = out->nDimension;
  THByteTensor* in = reinterpret_cast<THByteTensor*>(
      luaT_checkudata(L, 2, "torch.ByteTensor"));
  unsigned char* in_data = THByteTensor_data(in);
  const double accuracy = static_cast<double>(lua_tonumber(L, 3));

  // Hacky code to figure out what type 'real' is at runtime. This really should
  // be template specialization (so it's compiled in at runtime).
  real dummy;
  static_cast<void>(dummy);  // Silence compiler warnings.
  zfp_type type;
  if (typeid(dummy) == typeid(float)) {
    type = zfp_type_float;
  } else if (typeid(dummy) == typeid(double)) {
    type = zfp_type_double;
  } else {
    luaL_error(L, "Output type must be double or float.");
  }

  // Allocate meta data for the array.
  zfp_field* field;
  uint32_t dim_zfp;
  if (dim == 1) {
    field = zfp_field_1d(out_data, type, out->size[0]);
    dim_zfp = 1;
  } else if (dim == 2) {
    field = zfp_field_2d(out_data, type, out->size[1], out->size[0]);
    dim_zfp = 2;
  } else if (dim == 3) {
    field = zfp_field_3d(out_data, type, out->size[2], out->size[1],
                         out->size[0]);
    dim_zfp = 3;
  } else {
    // ZFP only allows up to 3D tensors, so we'll have to treat the input
    // tensor as a concatenated 3D tensor. This will affect compression ratios
    // but there's not much we can do about this.
    uint32_t sizez = 1;
    for (uint32_t i = 0; i < dim - 2; i++) {
      sizez *= out->size[i];
    }
    uint32_t sizey = out->size[dim - 2];
    uint32_t sizex = out->size[dim - 1];
    field = zfp_field_3d(out_data, type, sizex, sizey, sizez);
    dim_zfp = 3;
  }
  
  // Allocate meta data for the compressed stream.
  zfp_stream* zfp = zfp_stream_open(NULL);

  // Set stream compression mode and parameters.
  zfp_stream_set_accuracy(zfp, accuracy, type);

  // Get buffer for compressed data.
  void* buffer = reinterpret_cast<void*>(in_data);

  // Associate bit stream with allocated buffer.
  const uint32_t bufsize = in->size[0];
  bitstream* stream = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  
  // Compress entire array.
  const int ret = zfp_decompress(zfp, field);

  // Clean up.
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);

  if (!ret) { 
    luaL_error(L, "ZFP decompression failed!");
  }

  return 0;  // Recall: number of lua return items.
}

static const struct luaL_Reg torchzfp_(Main__) [] = {
  {"compress", torchzfp_(Main_compress)},
  {"decompress", torchzfp_(Main_decompress)},
  {NULL, NULL}
};

void torchzfp_(Main_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, torchzfp_(Main__), "torchzfp");
}

#endif
