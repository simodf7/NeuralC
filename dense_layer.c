#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dense_layer.h"

void activate_dense_layer(void* self, float* input){

    struct dense_layer* l = (struct dense_layer*) self;

    for(int i=0; i<l->n_neurons; i++){
        activate_neuron(l->neurons[i], input);
        l->base.output[i] = l->activation_function(l->neurons[i]->output); 
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

    for(int i=0; i< l->n_neurons; i++){
        destroy_neuron(l->neurons[i]); 
    }

    free(l->neurons); 
    free(l->base.output); 
    free(l); 
}




struct dense_layer* create_dense_layer(int n_neurons, int n_inputs_per_neuron, float (*func)(float)){

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


    l->n_neurons = n_neurons; 
    l->n_inputs_per_neuron = n_inputs_per_neuron; 
    l->activation_function = func; 


    l->neurons = malloc(n_neurons*sizeof(struct neuron*)); 
    if(!l->neurons){
        #ifdef DEBUG 
            printf("allocazione neuroni nel layer non andata a buon fine\n"); 
        #endif
        free(l);
        return NULL; 
    }

    l->base.output = malloc(n_neurons*sizeof(float)); 
    if(!l->base.output){
        #ifdef DEBUG 
            printf("allocazione output nel layer non andata a buon fine\n"); 
        #endif
        free(l->neurons); 
        free(l);
        return NULL; 
    }

    for(int i=0; i<n_neurons; i++){
        l->neurons[i] = create_neuron(n_inputs_per_neuron);
        if(!l->neurons[i]){
            #ifdef DEBUG 
                printf("Creazione %d-esimo neurone del layer non andata a buon fine\n", i); 
            #endif 
            for(int j=0; j<i; j++){
                free(l->neurons[j]); 
            }
            free(l->base.output); 
            free(l->neurons); 
            free(l); 
            return NULL; 
        }

    }


    #ifdef DEBUG 
        printf("Layer Denso creato con %d neuroni aventi %d input.\n", n_neurons, n_inputs_per_neuron); 
    #endif 
    return l;
}


void initialize_weights(struct dense_layer* l){

    for(int i=0; i<l->n_neurons; i++){
        float* w = initial_weights(l->n_inputs_per_neuron); 
        assign_weights(l->neurons[i], w); 
        free(w); 
    }

    #ifdef DEBUG 
        printf("Neuroni del layer inizializzati con pesi randomici.\n");    
    #endif 
}






