#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "softmax_layer.h"
#include <math.h>


void activate_softmax_layer(void* self, float* input){

    //casting 
    struct softmax_layer* s = (struct softmax_layer*) self; 

    float sum = 1e-8; 
    for(int i=0; i<s->n_inputs; i++){
        s->base.output[i] = expf(input[i]); 
        sum += s->base.output[i];
    }

    for(int i=0; i<s->n_inputs; i++){
        s->base.output[i] /= sum; 
    }


    #ifdef DEBUG 
        printf("Softmax Layer attivato. Output disponibili.\n");
    #endif 
}


void destroy_softmax_layer(void* self){

    struct softmax_layer* s = (struct softmax_layer*) self; 

    free(s->base.output); 
    free(s); 
}



struct softmax_layer* create_softmax_layer(int n_inputs){

    struct softmax_layer* s = malloc(sizeof(struct softmax_layer)); 
    if(!s){
        #ifdef DEBUG 
            printf("Allocazione softmax non andata a buon fine\n"); 
        #endif 
        return NULL; 
    }

    s->base.type = LAYER_SOFTMAX; 
    s->base.activate_layer = activate_softmax_layer; 
    s->base.destroy_layer = destroy_softmax_layer; 
    
    s->n_inputs = n_inputs; 

    s->base.output = malloc(n_inputs*sizeof(float)); 
    if(!s->base.output){
        #ifdef DEBUG 
            printf("allocazione output nel layer non andata a buon fine\n"); 
        #endif
        free(s);
        return NULL; 
    }

    #ifdef DEBUG
        printf("Layer softmax creato.\n"); 
    #endif

    return s; 
}





