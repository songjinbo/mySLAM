#添加opencv和PCL的依赖库
FIND_PACKAGE(PCL REQUIRED COMPONENTS common io visualization filters)
FIND_PACKAGE(OpenCV REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})

#添加g2o和csparse的依赖库
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
SET(G2O_ROOT /usr/local/include/g2o)
FIND_PACKAGE(G2O REQUIRED)

FIND_PACKAGE(CSparse REQUIRED)
INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

ADD_LIBRARY(stereoMatching stereoMatching.cpp)
TARGET_LINK_LIBRARIES(stereoMatching ${OpenCV_LIBS})

ADD_LIBRARY(generatePointCloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES(generatePointCloud ${OpenCV_LIBS} ${PCL_LIBRARY_DIRS}) 

ADD_LIBRARY(motionEstimation motionEstimation.cpp)
TARGET_LINK_LIBRARIES(motionEstimation ${OpenCV_LIBS} ${PCL_LIBRARY_DIRS}) 

ADD_EXECUTABLE(slam main.cpp)
TARGET_LINK_LIBRARIES(slam ${OpenCV_LIBS} ${PCL_LIBRARIES}
	generatePointCloud stereoMatching motionEstimation
	g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})
