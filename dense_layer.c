#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dense_layer.h"
#include "activation_function.h"

void activate_dense_layer(void* self, float* input){

    struct dense_layer* l = (struct dense_layer*) self;


    for(int i=0; i<l->n_neurons; i++){  // per ogni neurone
        float out = 0.0; 
        for(int j=0; j<l->base.n_inputs; j++){  // per ogni input
            out += l->weights[i*l->base.n_inputs + j]*input[j]; 
        }

        struct activation_output act = l->activation_function(out);  
        l->base.output[i] = act.out;   
        
        for(int j=0; j<l->base.n_inputs; j++){ 
            l->delta_weights[i*l->base.n_inputs + j] = act.der * input[j]; 
            l->delta_input[i*l->base.n_inputs + j] = act.der * l->weights[i*l->base.n_inputs + j];  // contributo di ogni neurone i all'input j
        }
        
        l->delta_bias[i] = act.der; 

    }


    #ifdef DEBUG 
        printf("Dense Layer attivato. Output disponibili: ");
        for(int i=0; i< l->n_neurons; i++){
            printf("%f ", l->base.output[i]);
        }
        printf("\n"); 
    #endif 
}


void destroy_dense_layer(void* self){

    struct dense_layer* l = (struct dense_layer*) self;

    free(l->delta_input);
    free(l->delta_bias); 
    free(l->delta_weights); 
    free(l->weights);
    free(l->base.output);
    free(l);
    
}




struct dense_layer* create_dense_layer(int n_neurons, int n_inputs_per_neuron, struct activation_output (*func)(float)){

    struct dense_layer* l = malloc(sizeof(struct dense_layer)); 
    if(!l){
        #ifdef DEBUG 
            printf("Allocazione layer non andata a buon fine\n"); 
        #endif 
        return NULL; 
    }

    l->base.type = LAYER_DENSE; 
    l->base.activate_layer = activate_dense_layer; 
    l->base.destroy_layer = destroy_dense_layer; 
    l->base.n_inputs = n_inputs_per_neuron; 
    
    l->n_neurons = n_neurons; 
    l->activation_function = func; 

    l->weights = malloc(n_neurons*n_inputs_per_neuron*sizeof(float)); 
    if(!l->weights){
        #ifdef DEBUG 
            printf("allocazione matrice dei pesi non andata a buon fine\n"); 
        #endif 
        free(l);
        return NULL; 
    }


    l->base.output = malloc(n_neurons*sizeof(float)); 
    if(!l->base.output){
        #ifdef DEBUG 
            printf("allocazione output nel layer non andata a buon fine\n"); 
        #endif
        free(l->weights);
        free(l);
        return NULL; 
    }

    l->delta_weights = malloc(n_neurons*n_inputs_per_neuron*sizeof(float)); 
    if(!l->delta_weights){
        #ifdef DEBUG 
            printf("allocazione matrice dei local gradient ai pesi non andata a buon fine\n"); 
        #endif 
        free(l->weights);
        free(l->base.output);
        free(l);
        return NULL; 
    }

    l->delta_bias = malloc(n_neurons*sizeof(float)); 
    if(!l->delta_bias){
        #ifdef DEBUG 
            printf("allocazione matrice dei local gradient ai bias non andata a buon fine\n"); 
        #endif 
        free(l->delta_weights); 
        free(l->weights);
        free(l->base.output);
        free(l);
        return NULL; 
    }


    l->delta_input = malloc(n_neurons*n_inputs_per_neuron*sizeof(float)); 
    if(!l->delta_input){
        #ifdef DEBUG 
            printf("allocazione matrice dei local gradient agli input non andata a buon fine\n"); 
        #endif 
        free(l->delta_bias); 
        free(l->delta_weights); 
        free(l->weights);
        free(l->base.output);
        free(l);
        return NULL; 
    }


    #ifdef DEBUG 
        printf("Layer Denso creato con %d neuroni aventi %d input.\n", n_neurons, n_inputs_per_neuron); 
    #endif 
    return l;
}


void initialize_weights(struct dense_layer* l){

    for(int i=0; i<l->n_neurons; i++){
        float* w = initial_weights(l->base.n_inputs); 
        for(int j=0; j< l->base.n_inputs; j++){
            l->weights[i*l->base.n_inputs + j] = w[j];
        }
        free(w); 
    }

    #ifdef DEBUG 
        printf("Neuroni del layer inizializzati con pesi randomici.\n");    
    #endif 
}






