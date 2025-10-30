#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "loss.h"

// i'm designing a classification NN so i will implement just relevant losses


struct loss* cross_entropy_loss(float* output, int n_class, int index_correct_class){
    struct loss* l = malloc(sizeof(struct loss)); 
    if(!l){
        printf("Error in loss allocation.\n"); 
        return NULL; 
    }

    l->local_gradient = malloc(n_class*sizeof(float)); 
    if(!l->local_gradient){
        printf("Error in local gradient loss allocation.\n"); 
        free(l);
        return NULL; 
    }

    
    l->output = -log10f(output[index_correct_class]); 
    for(int i=0; i<n_class; i++){
        if(i != index_correct_class){
            l->local_gradient[i] = 0.0f; 
        }
        else
            l->local_gradient[i] = -1.0f/l->output;
    } 
}
