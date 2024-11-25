# CMake generated Testfile for 
# Source directory: /home/myl/RTNNproject/rt_filter/deps/cxxopts/test
# Build directory: /home/myl/RTNNproject/rt_filter/deps/cxxopts/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[options]=] "options_test")
set_tests_properties([=[options]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;23;add_test;/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;0;")
add_test([=[find-package-test]=] "/usr/local/bin/ctest" "-C" "--build-and-test" "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/find-package-test" "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/find-package-test" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/make" "--build-options" "-DCMAKE_CXX_COMPILER=/usr/bin/c++" "-DCMAKE_BUILD_TYPE=" "-Dcxxopts_DIR=/home/myl/RTNNproject/rt_filter/deps/cxxopts")
set_tests_properties([=[find-package-test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;26;add_test;/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;0;")
add_test([=[add-subdirectory-test]=] "/usr/local/bin/ctest" "-C" "--build-and-test" "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/add-subdirectory-test" "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/add-subdirectory-test" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/make" "--build-options" "-DCMAKE_CXX_COMPILER=/usr/bin/c++" "-DCMAKE_BUILD_TYPE=")
set_tests_properties([=[add-subdirectory-test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;40;add_test;/home/myl/RTNNproject/rt_filter/deps/cxxopts/test/CMakeLists.txt;0;")
