project(SchwarzwaldService)

set(SOURCE_FILES main.cpp)

add_executable(SchwarzwaldService ${SOURCE_FILES})

target_include_directories(SchwarzwaldService
                           PRIVATE ${CMAKE_SOURCE_DIR}/lib/cpp-httplib)

if(UNIX)
  target_link_libraries(SchwarzwaldService PUBLIC SchwarzwaldCore pthread)
else()
  target_link_libraries(SchwarzwaldService PUBLIC SchwarzwaldCore)
endif()
