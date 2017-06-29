# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/song/SLAM/StereoMatching

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/song/SLAM/StereoMatching

# Include any dependencies generated for this target.
include src/CMakeFiles/motionEstimation.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/motionEstimation.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/motionEstimation.dir/flags.make

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o: src/CMakeFiles/motionEstimation.dir/flags.make
src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o: src/motionEstimation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/song/SLAM/StereoMatching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o"
	cd /home/song/SLAM/StereoMatching/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o -c /home/song/SLAM/StereoMatching/src/motionEstimation.cpp

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/motionEstimation.dir/motionEstimation.cpp.i"
	cd /home/song/SLAM/StereoMatching/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/song/SLAM/StereoMatching/src/motionEstimation.cpp > CMakeFiles/motionEstimation.dir/motionEstimation.cpp.i

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/motionEstimation.dir/motionEstimation.cpp.s"
	cd /home/song/SLAM/StereoMatching/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/song/SLAM/StereoMatching/src/motionEstimation.cpp -o CMakeFiles/motionEstimation.dir/motionEstimation.cpp.s

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.requires:

.PHONY : src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.requires

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.provides: src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/motionEstimation.dir/build.make src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.provides.build
.PHONY : src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.provides

src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.provides.build: src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o


# Object files for target motionEstimation
motionEstimation_OBJECTS = \
"CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o"

# External object files for target motionEstimation
motionEstimation_EXTERNAL_OBJECTS =

lib/libmotionEstimation.a: src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o
lib/libmotionEstimation.a: src/CMakeFiles/motionEstimation.dir/build.make
lib/libmotionEstimation.a: src/CMakeFiles/motionEstimation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/song/SLAM/StereoMatching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../lib/libmotionEstimation.a"
	cd /home/song/SLAM/StereoMatching/src && $(CMAKE_COMMAND) -P CMakeFiles/motionEstimation.dir/cmake_clean_target.cmake
	cd /home/song/SLAM/StereoMatching/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/motionEstimation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/motionEstimation.dir/build: lib/libmotionEstimation.a

.PHONY : src/CMakeFiles/motionEstimation.dir/build

src/CMakeFiles/motionEstimation.dir/requires: src/CMakeFiles/motionEstimation.dir/motionEstimation.cpp.o.requires

.PHONY : src/CMakeFiles/motionEstimation.dir/requires

src/CMakeFiles/motionEstimation.dir/clean:
	cd /home/song/SLAM/StereoMatching/src && $(CMAKE_COMMAND) -P CMakeFiles/motionEstimation.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/motionEstimation.dir/clean

src/CMakeFiles/motionEstimation.dir/depend:
	cd /home/song/SLAM/StereoMatching && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/song/SLAM/StereoMatching /home/song/SLAM/StereoMatching/src /home/song/SLAM/StereoMatching /home/song/SLAM/StereoMatching/src /home/song/SLAM/StereoMatching/src/CMakeFiles/motionEstimation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/motionEstimation.dir/depend

