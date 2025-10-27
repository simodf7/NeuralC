#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>

struct activation_output{
    float out; 
    float der; // derivata 
}; 

struct activation_output reLU(float); 
struct activation_output sigmoid(float); 

#endif 
