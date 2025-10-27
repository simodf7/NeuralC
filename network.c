#include <stdio.h>
#include <stdlib.h>
#include "network.h"

struct network* create_network(int n_layers, int n_outputs){

    struct network* n = malloc(sizeof(struct network)); 
    if(!n){
        #ifdef DEBUG
            printf("Allocazione della rete non andata a buon fine.\n"); 
        #endif 
        return NULL; 
    }

    n->n_layers = n_layers; 
    n->n_outputs = n_outputs;

    n->layers = malloc(sizeof(struct layer*)*n_layers); 
    if(!n->layers){
        #ifdef DEBUG
            printf("Allocazione del vettore dei puntatori ai layer non andata a buon fine.\n"); 
        #endif 
        free(n); 
        return NULL;
    }


    n->output = malloc(sizeof(float)*n_outputs); 
    if(!n->output){
        #ifdef DEBUG
            printf("Allocazione del vettore delle uscite della rete non andata a buon fine.\n"); 
        #endif 
        free(n->layers); 
        free(n); 
        return NULL;
    }

    return n; 

}

void add_layer(struct network* n, struct layer* l, int pos){
    if(pos >= 0 && pos < n->n_layers){
        n->layers[pos] = l; 
    }
    else{
        printf("Layer non disponibile.\n"); 
    }
}

void forward(struct network* n, float* input){

    n->layers[0]->activate_layer(n->layers[0], input); 

    for(int i=1; i< n->n_layers; i++){
        n->layers[i]->activate_layer(n->layers[i], n->layers[i-1]->output); 
    }

    for(int i=0; i<n->n_outputs; i++){
        n->output[i] = n->layers[n->n_layers-1]->output[i]; 
    }
    
}

void backward(struct network* n, float loss){

    


}





void initialize_network(struct network* n){

    for(int i=0; i< n->n_layers; i++){
        if(n->layers[i]->type == LAYER_DENSE){  // da cambiare
            initialize_weights((struct dense_layer*) n->layers[i]);
        } 
    }

}


void destroy_network(struct network* n){
    for(int i=0; i< n->n_layers; i++){
        n->layers[i]->destroy_layer(n->layers[i]); 
    }

    free(n->layers); 
    free(n);
}



