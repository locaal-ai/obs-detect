include(FetchContent)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  URL_HASH SHA256=1CCAABBFE870F60AF3D6A519C53E09F3DCF630207321DFFA553564A8E75C4FC8)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
  # add eigen to include path
  include_directories(${eigen_SOURCE_DIR})
endif()
