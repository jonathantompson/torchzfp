CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

set(BUILD_SHARED_LIBS FALSE)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/inc")
FILE(GLOB ZFP_HDRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h")
MESSAGE("Found ZFP headers: ${ZFP_HDRS}")
FILE(GLOB ZFP_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
MESSAGE("Found ZFP source: ${ZFP_SRC}")

ADD_DEFINITIONS(-fPIC)

IF (${BUILD_SHARED_LIBS})
  MESSAGE("Building shared libraries")
  ADD_DEFINITIONS(-DZFP_DLL)
  ADD_LIBRARY(zfp SHARED ${ZFP_SRC} ${ZFP_HDRS})
ELSE()
  MESSAGE("Building static libraries")
  ADD_LIBRARY(zfp ${ZFP_SRC} ${ZFP_HDRS})
ENDIF()

#set vars as in FindZFP.cmake
SET(ZFP_FOUND 1                                        CACHE BOOL "" FORCE)
SET(ZFP_INCLUDE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/inc" CACHE STRING "" FORCE)
SET(ZFP_INCLUDE_DIRS "${ZFP_INCLUDE_DIR}"              CACHE STRING "" FORCE)
SET(ZFP_LIBRARY      zfp                               CACHE STRING "" FORCE)
SET(ZFP_LIBRARIES    stdc++ ${ZFP_LIBRARY}             CACHE STRING "" FORCE)
