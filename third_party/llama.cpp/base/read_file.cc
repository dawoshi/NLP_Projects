/* Copyright 2023, The GenC Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include <fstream>
#include <iterator>
#include <string>

#include "third_party/llama.cpp/base/read_file.h"

namespace genc {

std::string ReadFile(std::string filename) {
  std::ifstream file {filename};
  std::string content {
    std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  return content;
}

}  // namespace genc
