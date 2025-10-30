#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"

struct softmax_layer{
    struct layer base; 
    int n_inputs; 
    float* local_gradient; 
}; 


struct softmax_layer* create_softmax_layer(int n_inputs); 
void activate_softmax_layer(void* self, float* input);  // deve essere nella stessa forma della funzione nella struct layer
void destroy_softmax_layer(void* self); 
#endif



