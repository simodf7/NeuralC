#ifndef SGD_H
#define SGD_H

#include "optimizer.h"


struct optimizer* sgd(float, struct network*);
void sgd_update_weights(struct optimizer*); 

#endif 