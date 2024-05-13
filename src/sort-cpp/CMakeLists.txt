
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(sort-cpp ${SOURCE_FILES})

target_include_directories(sort-cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sort-cpp PRIVATE OpenCV)
