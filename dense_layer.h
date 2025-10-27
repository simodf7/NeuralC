#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H 
#include "layer.h" 

struct dense_layer{
    struct layer base; 
    int n_neurons; 
    float* weights; // neuroni x input
    float* delta_weights; // local_gradient for weights; // to calculate gradient to update weight
    float* delta_bias; // local gradient for bias;; // to calculate graident to update bias
    float* delta_input; // local_gradient for input -- to calculate gradient to flow upstream 
    struct activation_output (*activation_function)(float); 
};


struct dense_layer* create_dense_layer(int n_neurons, int n_inputs_per_neuron, struct activation_output (*func)(float)); 
void initialize_weights(struct dense_layer* l); 
void activate_dense_layer(void* self, float* input); 
void destroy_dense_layer(void* self);
#endif 