#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include "gflags.h"
#include "base/logging.h"
#include "chinese_error_correction/t52csc/onnx-cpp/model/t5_error_correction.h"

using namespace error_correction;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string tmp;
  T5ErrorCorrection detector;
  detector.Init();
  
  struct timeval t1, t2; 
  std::vector<std::string> courps;
  std::ifstream infile;
  infile.open("./test.txt");
  if (!infile.is_open())
      LOG(INFO) << "open file failure";
  while (!infile.eof()) {
    std::string line;
    while (getline(infile, line)) {
      courps.push_back(line);
    }   
  }   
  infile.close();
  LOG(INFO) << "test courps len count: " << courps.size();
  std::ofstream outfile;
  outfile.open("./msra_test.txt");
  if(!outfile.is_open()) {
      return -1;
  }
  double total_time = 0.0;
  for (size_t i = 0; i < courps.size(); ++i) {
    gettimeofday(&t1, NULL);
    std::string tmp;
    detector.predict(courps[i], &tmp);
    gettimeofday(&t2, NULL);
    total_time += (t2.tv_sec - t1.tv_sec) +
      (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    outfile << tmp;
    if(!tmp.empty()) {
        outfile << "\n";
    }
  }
  outfile.close();  
  LOG(INFO) << "Totle run Time : " << total_time * 1000.0 << "ms";
  return 0;
}
