#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#include "neuron.h"

// aggiungere controlli if(!n) es. 
// aggiungere if debug per togliere gli if


struct neuron* create_neuron(int n_inputs){
    struct neuron* n = malloc(sizeof(struct neuron));
    if (!n){ 
        #ifdef DEBUG 
            printf("Creazione Neurone non andata a buon fine"); 
        #endif 
        return NULL; 
    }

    n->weights = malloc(n_inputs*sizeof(float)); 
    if(!n->weights){
        #ifdef DEBUG 
            printf("Creazione del vettore di pesi non andata a buon fine"); 
        #endif 
        free(n); 
        return NULL; 
    }    


    n->n_inputs = n_inputs;
    return n;
}



void assign_weights(struct neuron* n, float* w){
    
    for(int i=0; i< n->n_inputs; i++){
        n->weights[i] = w[i]; 
    }
    n->bias = w[n->n_inputs]; 
    return; 
}


void activate_neuron(struct neuron* n, float* input){

    float output = 0.0; 

    for(int i = 0; i< n->n_inputs; i++){
        output += n->weights[i]*input[i]; 
    }

    output += n->bias;

    n->output = output; 

   
    #ifdef DEBUG
        printf("Output Somma pesata: %f\n", output);
    #endif
    
    // manca funzione di attivazione 
}


float* initial_weights(int n_inputs){
    
    float* w = malloc( (n_inputs + 1) * sizeof(float));

    for(int i=0; i< n_inputs; i++){
        w[i] = ((float) rand() / RAND_MAX) - 0.5f; // inizilamente assegniamo un peso tra -0.5 e 0.5
    }   

    w[n_inputs] =  ((float) rand() / RAND_MAX) - 0.5f; 

    return w; 
}


void destroy_neuron(struct neuron* n){
    free(n->weights); 
    free(n); 
}




