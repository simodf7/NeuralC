#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "network.h"

struct optimizer{   
    float learning_rate; // da aggiungere altro     
    struct network* net; 
    void (*update_weights)(struct optimizer* self);
};



#endif 