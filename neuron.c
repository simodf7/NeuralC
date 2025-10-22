#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#define MAX_INPUT 4

struct neuron{
    float weights[MAX_INPUT]; 
    float bias; 
    float (*activation_function)(float);  // puntatore a funzione che prende float e restituisce float
    float output; 
    float local_gradient; // da capire
};



struct neuron* create_neuron(float (*func)(float)){
    struct neuron* n = malloc(sizeof(struct neuron));
    if (!n){ 
        printf("Creazione Neurone non andata a buon fine"); 
        return NULL; 
    }

    n->activation_function = func;
    return n;
}



void assign_weights(struct neuron* n, float w[MAX_INPUT+1]){
    
    for(int i=0; i<MAX_INPUT; i++){
        n->weights[i] = w[i]; 
    }
    n->bias = w[MAX_INPUT]; 
    return; 
}


void activate_neuron(struct neuron* n, float input[MAX_INPUT]){

    float output = 0.0; 

    for(int i = 0; i<MAX_INPUT; i++){
        output += n->weights[i]*input[i]; 
    }

    output += n->bias;

    printf("Output Somma pesata: %f\n", output);

    n->output = n->activation_function(output); 
}

float reLU(float output){
    if(output < 0){
        return 0.0; 
    }
    else
        return output; 
}

float* initial_weights(){
    
    float* w = malloc( (MAX_INPUT + 1) * sizeof(float));

    for(int i=0; i<MAX_INPUT; i++){
        w[i] = ((float) rand() / RAND_MAX) - 0.5f; // inizilamente assegniamo un peso tra -0.5 e 0.5
    }   

    w[MAX_INPUT] =  ((float) rand() / RAND_MAX) - 0.5f; 

    return w; 
}


int main(void){
    srand((unsigned int) time(NULL));

    printf("Creazione Neurone\n"); 

    struct neuron* n1 = create_neuron(reLU); 
    float* w = initial_weights(); 

    printf("Pesi iniziali: "); 

    for(int i=0; i<MAX_INPUT; i++){
        printf("%f ", w[i]); 
    }

    
    printf("\nAssegno Pesi\n");
    assign_weights(n1, w); 

    free(w); 

    float input[MAX_INPUT] = {2.1, 3.4, -1.2, 0.0}; 

    activate_neuron(n1, input); 
    printf("Output del neurone: %f \n", n1->output);     

    free(n1);
    return 0; 
}



