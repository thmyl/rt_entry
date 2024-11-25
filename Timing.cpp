#include "Timing.h"

std::stack<TimingHelper> Timing::timingStack;
unsigned int Timing::startCounter = 0;
unsigned int Timing::stopCounter = 0;