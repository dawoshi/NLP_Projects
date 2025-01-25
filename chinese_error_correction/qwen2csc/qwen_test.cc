#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include "gflags.h"
#include "base/logging.h"
#include "chinese_error_correction/qwen2csc/llama_cpp.h"

using namespace llama_cpp;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string tmp;
  LLamaCpp detector;
  detector.Init();
  
  struct timeval t1, t2; 
  std::vector<std::string> courps;
  std::ifstream infile;
  infile.open("./test.txt");
  if (!infile.is_open())
      LOG_TEE("open file failure");
  while (!infile.eof()) {
    std::string line;
    while (getline(infile, line)) {
      courps.push_back(line);
    }   
  }   
  infile.close();
  LOG_TEE("test courps len count: ");
  std::ofstream outfile;
  outfile.open("./msra_test.txt");
  if(!outfile.is_open()) {
      return -1;
  }
  double total_time = 0.0;

  std::string k_system =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:)";

  gettimeofday(&t1, NULL);
  detector.infer(k_system, courps);
  gettimeofday(&t2, NULL);
  total_time += (t2.tv_sec - t1.tv_sec) +
    (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
  outfile.close();  
  LOG_TEE("Totle run Time : ");
  // LOG_TEE(total_time * 1000.0);
  return 0;
}
