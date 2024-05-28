include(FetchContent)

set(CUSTOM_OPENCV_URL
    ""
    CACHE STRING "URL of a downloaded OpenCV static library tarball")

set(CUSTOM_OPENCV_HASH
    ""
    CACHE STRING "Hash of a downloaded OpenCV staitc library tarball")

if(CUSTOM_OPENCV_URL STREQUAL "")
  set(USE_PREDEFINED_OPENCV ON)
else()
  if(CUSTOM_OPENCV_HASH STREQUAL "")
    message(FATAL_ERROR "Both of CUSTOM_OPENCV_URL and CUSTOM_OPENCV_HASH must be present!")
  else()
    set(USE_PREDEFINED_OPENCV OFF)
  endif()
endif()

if(USE_PREDEFINED_OPENCV)
  set(OpenCV_VERSION "v4.9.0-1")
  set(OpenCV_BASEURL "https://github.com/obs-ai/obs-backgroundremoval-dep-opencv/releases/download/${OpenCV_VERSION}")

  if(${CMAKE_BUILD_TYPE} STREQUAL Release OR ${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
    set(OpenCV_BUILD_TYPE Release)
  else()
    set(OpenCV_BUILD_TYPE Debug)
  endif()

  if(APPLE)
    if(OpenCV_BUILD_TYPE STREQUAL Debug)
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-macos-${OpenCV_VERSION}-Debug.tar.gz")
      set(OpenCV_HASH SHA256=BE85C8224F71C52162955BEE4EC9FFBE41CBED636D7989843CA75AD42657B121)
    else()
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-macos-${OpenCV_VERSION}-Release.tar.gz")
      set(OpenCV_HASH SHA256=5DB4FCFBD8C7CDBA136657B4D149821A670DF9A7C71120F5A4D34FA35A58D07B)
    endif()
  elseif(MSVC)
    if(OpenCV_BUILD_TYPE STREQUAL Debug)
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-windows-${OpenCV_VERSION}-Debug.zip")
      set(OpenCV_HASH SHA256=0A1BBC898DCE5F193427586DA84D7A34BBB783127957633236344E9CCD61B9CE)
    else()
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-windows-${OpenCV_VERSION}-Release.zip")
      set(OpenCV_HASH SHA256=56A5E042F490B8390B1C1819B2B48C858F10CD64E613BABBF11925A57269C3FA)
    endif()
  else()
    if(OpenCV_BUILD_TYPE STREQUAL Debug)
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-linux-${OpenCV_VERSION}-Debug.tar.gz")
      set(OpenCV_HASH SHA256=840A7D80B661CFF7B7300272A2A2992D539672ECECA01836B85F68BD8CAF07F5)
    else()
      set(OpenCV_URL "${OpenCV_BASEURL}/opencv-linux-${OpenCV_VERSION}-Release.tar.gz")
      set(OpenCV_HASH SHA256=73652C2155B477B5FD95FCD8EA7CE35D313543ECE17BDFA3A2B217A0239D74C6)
    endif()
  endif()
else()
  set(OpenCV_URL "${CUSTOM_OPENCV_URL}")
  set(OpenCV_HASH "${CUSTOM_OPENCV_HASH}")
endif()

FetchContent_Declare(
  opencv
  URL ${OpenCV_URL}
  URL_HASH ${OpenCV_HASH})
FetchContent_MakeAvailable(opencv)

add_library(OpenCV INTERFACE)
if(MSVC)
  target_link_libraries(
    OpenCV
    INTERFACE ${opencv_SOURCE_DIR}/x64/vc17/staticlib/opencv_imgproc490.lib
              ${opencv_SOURCE_DIR}/x64/vc17/staticlib/opencv_core490.lib
              ${opencv_SOURCE_DIR}/x64/vc17/staticlib/opencv_video490.lib
              ${opencv_SOURCE_DIR}/x64/vc17/staticlib/zlib.lib)
  target_include_directories(OpenCV SYSTEM INTERFACE ${opencv_SOURCE_DIR}/include)
else()
  target_link_libraries(
    OpenCV INTERFACE ${opencv_SOURCE_DIR}/lib/libopencv_imgproc.a ${opencv_SOURCE_DIR}/lib/libopencv_core.a
                     ${opencv_SOURCE_DIR}/lib/libopencv_video.a ${opencv_SOURCE_DIR}/lib/opencv4/3rdparty/libzlib.a)
  target_include_directories(OpenCV SYSTEM INTERFACE ${opencv_SOURCE_DIR}/include/opencv4)
endif()
