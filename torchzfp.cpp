
#include <TH.h>
#include <luaT.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define torchzfp_(NAME) TH_CONCAT_3(torchzfp_, Real, NAME)

#ifdef max
#undef max
#endif
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#ifdef min
#undef min
#endif
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#include "generic/torchzfp.cpp"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libtorchzfp(lua_State *L) {
  torchzfp_FloatMain_init(L);
  torchzfp_DoubleMain_init(L);

  luaL_register(L, "torchzfp.float", torchzfp_FloatMain__);
  luaL_register(L, "torchzfp.double", torchzfp_DoubleMain__); 
  
  return 1;
}
