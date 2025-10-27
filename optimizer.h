#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "network.h"

struct optimizer{   
    float learning_rate; // da aggiungere altro     
    void (*update_weight(void* self, struct network*));
};



#endif 