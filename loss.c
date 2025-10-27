#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "loss.h"

// i'm designing a classification NN so i will implement just relevant losses


struct loss* cross_entropy_loss(float* output, int index_correct_class){
    struct loss* l = malloc(sizeof(struct loss)); 
    if(!l){
        printf("Error in loss allocation.\n"); 
        return NULL; 
    }
    
    l->output = -log10f(output[index_correct_class]); 
    l->local_gradient = -(1/l->output); 
}
