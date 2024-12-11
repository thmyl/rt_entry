#pragma once

#include <bits/stdc++.h>
#define OUTFILE "out.txt"

struct TimingHelper{
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::string name;
};

class Timing{
public:
  static std::stack<TimingHelper> timingStack;
  static unsigned int startCounter;
  static unsigned int stopCounter;
  static void reset(){
    while(!timingStack.empty())
      timingStack.pop();
    startCounter = 0;
    stopCounter = 0;
  }

  static void startTiming(const std::string& name = std::string("")){
    TimingHelper h;
    h.start = std::chrono::high_resolution_clock::now();
    h.name = name;
    timingStack.push(h);
    startCounter++;
  }

  static double stopTiming(int print = 1){
    if(!Timing::timingStack.empty()){
      Timing::stopCounter++;
      std::chrono::time_point<std::chrono::high_resolution_clock> stop = std::chrono::high_resolution_clock::now();
			TimingHelper h = Timing::timingStack.top();
			Timing::timingStack.pop();
			std::chrono::duration<double> elapsed_seconds = stop - h.start;
			double t = elapsed_seconds.count() * 1000.0;

      if (print == 1)
				std::cout << "time " << h.name.c_str() << ": " << t << " ms\n" << std::flush;
      else if(print == 2){
        std::cout << "time " << h.name.c_str() << ": " << t << " ms\n" << std::flush;
        // std::ofstream outfile;
        // outfile.open(OUTFILE, std::ios_base::app);
        // outfile <<  "time " << h.name.c_str() << ": " << t << " ms\n" << std::flush;
        // outfile.close();
      }
			return t;
    }
    return 0;
  }
};