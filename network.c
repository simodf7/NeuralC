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

void backward(struct network* n, float* loss_grad){

    // softmax_layer 
    struct softmax_layer* s = (struct softmax_layer*) n->layers[n->n_layers-1];

    float* up = malloc(s->n_inputs * sizeof(float));   // vettore dei downstream da softmax uno per neurone
    if(!up){
        #ifdef DEBUG
        printf("Allocazione del vettore del grdiente upstream non andata a buon fine.\n"); 
        #endif 
        return NULL;
    }


    for(int i=0; i<s->n_inputs; i++){
        up[i] = 0.0f; 
        for(int j=0; j<s->n_inputs; j++){
            up[i] += (loss_grad[j] * s->local_gradient[i*s->n_inputs + j]); 
        }
    }
    

    struct dense_layer* d; 
    

    for(int i=1; i<n->n_layers-1; i++){  // devo escludere softmax
        d = (struct dense_layer*) n->layers[n->n_layers-1-i];
        float* up_next = malloc(d->base.n_inputs * sizeof(float)); 
        if(!up_next){
            #ifdef DEBUG
            printf("Allocazione del vettore del grdiente upstream non andata a buon fine.\n"); 
            #endif 
            return NULL;
        }

        for(int k=0; k<d->base.n_inputs; k++){
            up_next[k] = 0.0f; 
        }

        for(int j=0; j<d->n_neurons; j++){
            d->delta_bias[j] = d->delta_bias[j] * up[j];  // dL/out_neuron_i * dout_neuron_i/d_bias_i
            for(int k=0; k<d->base.n_inputs; k++){ 
                d->delta_weights[j*d->base.n_inputs + k] = d->delta_weights[j*d->base.n_inputs + k] * up[j];  // gradiente locale per dL/dz1
                up_next[k] += (d->delta_input[j*d->base.n_inputs + k] * up[j]);  // contributo di ogni neurone i all'input j
            }
        } 

        free(up);
        up = up_next; 
    }


    free(up); 
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



