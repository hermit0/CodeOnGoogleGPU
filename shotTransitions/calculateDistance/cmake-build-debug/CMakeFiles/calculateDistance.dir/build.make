# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /snap/clion/57/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/57/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/calculateDistance.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calculateDistance.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calculateDistance.dir/flags.make

CMakeFiles/calculateDistance.dir/main.cpp.o: CMakeFiles/calculateDistance.dir/flags.make
CMakeFiles/calculateDistance.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/calculateDistance.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calculateDistance.dir/main.cpp.o -c /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/main.cpp

CMakeFiles/calculateDistance.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calculateDistance.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/main.cpp > CMakeFiles/calculateDistance.dir/main.cpp.i

CMakeFiles/calculateDistance.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calculateDistance.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/main.cpp -o CMakeFiles/calculateDistance.dir/main.cpp.s

# Object files for target calculateDistance
calculateDistance_OBJECTS = \
"CMakeFiles/calculateDistance.dir/main.cpp.o"

# External object files for target calculateDistance
calculateDistance_EXTERNAL_OBJECTS =

calculateDistance: CMakeFiles/calculateDistance.dir/main.cpp.o
calculateDistance: CMakeFiles/calculateDistance.dir/build.make
calculateDistance: /usr/local/lib/libopencv_core.so
calculateDistance: /usr/local/lib/libopencv_videoio.so
calculateDistance: /usr/local/lib/libopencv_imgproc.so
calculateDistance: /usr/lib/x86_64-linux-gnu/libprotobuf.so
calculateDistance: /home/hermit/C3D-v1.1-openblas/build/lib/libcaffe.so
calculateDistance: /usr/lib/x86_64-linux-gnu/libboost_system.so
calculateDistance: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
calculateDistance: CMakeFiles/calculateDistance.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable calculateDistance"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calculateDistance.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calculateDistance.dir/build: calculateDistance

.PHONY : CMakeFiles/calculateDistance.dir/build

CMakeFiles/calculateDistance.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calculateDistance.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calculateDistance.dir/clean

CMakeFiles/calculateDistance.dir/depend:
	cd /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug /home/hermit/CodeOnGoogleGPU/shotTransitions/calculateDistance/cmake-build-debug/CMakeFiles/calculateDistance.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calculateDistance.dir/depend

