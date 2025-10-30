#include <stdio.h>
#include <stdlib.h>
#include "sgd.h"


void sgd_update_weights(struct optimizer* self){

    for(int i=0; i<self->net->n_layers-1; i++){
        struct dense_layer* d = (struct dense_layer*) self->net->layers[i]; 
        for(int j=0; j<d->n_neurons; j++){
            d->bias[j] -= self->learning_rate * d->delta_bias[j]; 
            for(int k=0; k<d->base.n_inputs; k++){
                d->weights[j*d->base.n_inputs + k] -= self->learning_rate * d->delta_weights[j*d->base.n_inputs + k]; 
            }
        } 
    }

}

struct optimizer* sgd(float learning_rate, struct network* n){

    struct optimizer* sgd = malloc(sizeof(struct optimizer)); 
    if(!sgd){
        #ifdef DEBUG
            printf("Allocazione di sgd non andata a buon fine.\n"); 
            #endif 
            return NULL;
    }

    sgd->learning_rate = learning_rate; 
    sgd->net = n; 
    sgd->update_weights = sgd_update_weights; 

    return sgd;
}


