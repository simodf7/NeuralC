// trying to implement a strategy pattern
#ifndef LAYER_H
#define LAYER_H

enum layer_type{
    LAYER_DENSE, 
    LAYER_SOFTMAX
};

struct layer{
    enum layer_type type; 
    float* output; 
    void (*activate_layer)(void* self, float* input); 
    void (*destroy_layer)(void* self); 
    // per il momento solo questo
}; 
#endif 




