#include "activation_function.h"

struct activation_output reLU(float output){

    struct activation_output a; 

    if(output <= 0){
        a.out = 0.0f; 
        a.der = 0.0f; 
    }
    else{
        a.out = output; 
        a.der = 1.0f; 
    }
    
    return a; 
}


struct activation_output sigmoid(float output){
    struct activation_output a;
    a.out = 1.0f /(1.0f +exp(-output)); 
    a.der = a.out * (1.0f - a.out); 
    return a;
}
