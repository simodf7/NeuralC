#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neuron.h"
#include "activation_function.h"

int main(void){
    srand((unsigned int) time(NULL));

    printf("Creazione Neurone\n"); 

    int n = 4; 

    struct neuron* n1 = create_neuron(reLU, n); 
    float* w = initial_weights(n); 

    printf("Pesi iniziali: "); 

    for(int i=0; i<n; i++){
        printf("%f ", w[i]); 
    }

    
    printf("\nAssegno Pesi\n");
    assign_weights(n1, w); 

    free(w); 

    float* input = malloc(n*sizeof(float)); 

    input[0] = 2.1; 
    input[1] = 3.4; 
    input[2] = -1.2;
    input[3] = 0.0; 


    activate_neuron(n1, input); 
    printf("Output del neurone: %f \n", n1->output);     

    free(input); 
    free(n1->weights); 
    free(n1);
    return 0; 
}
