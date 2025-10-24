#ifndef NETWORK_H
#define NETWORK_H
#include "dense_layer.h"
#include "softmax_layer.h"

struct network{
    int n_layers; 
    struct layer** layers; 
    int n_outputs; 
    float* output; 
};


struct network* create_network(int n_layers, int n_outputs);
void add_layer(struct network* n, struct layer* l, int pos);
void forward(struct network* n, float* input);
void initialize_network(struct network* n);
void destroy_network(struct network* n);
#endif