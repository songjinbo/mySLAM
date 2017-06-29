FIND_PACKAGE(PCL REQUIRED COMPONENTS common io visualization filters)
FIND_PACKAGE(OpenCV REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})

ADD_LIBRARY(stereoMatching stereoMatching.cpp)
TARGET_LINK_LIBRARIES(stereoMatching ${OpenCV_LIBS})

ADD_LIBRARY(generatePointCloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES(generatePointCloud ${OpenCV_LIBS} ${PCL_LIBRARY_DIRS}) 

ADD_LIBRARY(motionEstimation motionEstimation.cpp)
TARGET_LINK_LIBRARIES(motionEstimation ${OpenCV_LIBS} ${PCL_LIBRARY_DIRS}) 

ADD_EXECUTABLE(slam main.cpp)
TARGET_LINK_LIBRARIES(slam ${OpenCV_LIBS} ${PCL_LIBRARIES} generatePointCloud stereoMatching motionEstimation)
