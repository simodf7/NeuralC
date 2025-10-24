#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H 
#include "layer.h" 

struct dense_layer{
    struct layer base; 
    int n_neurons; 
    int n_inputs_per_neuron; 
    struct neuron** neurons;  // array di puntatori a neuroni
    float (*activation_function)(float); 
};


struct dense_layer* create_dense_layer(int n_neurons, int n_inputs_per_neuron, float (*func)(float)); 
void initialize_weights(struct dense_layer* l); 
void activate_dense_layer(void* self, float* input); 
void destroy_dense_layer(void* self);
#endif 