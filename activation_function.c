#include "activation_function.h"

float reLU(float output){
    if(output < 0){
        return 0.0; 
    }
    else
        return output; 
}


float sigmoid(float output){
    return 1/(1+exp(-output)); 
}
