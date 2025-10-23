#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "layer.h"
#include "activation_function.h"

int main(void) {
    srand((unsigned int) time(NULL));

    int n_inputs = 4;
    int n_neurons = 3;

    printf("Creazione layer...\n");

    struct layer* l1 = create_layer(n_neurons, n_inputs, reLU);
    if (!l1) {
        printf("Errore nella creazione del layer.\n");
        return 1;
    }

    initialize_weights(l1);

    // Input di prova
    float input[4] = {1.0f, -2.0f, 0.5f, 3.0f};

    printf("Attivazione layer...\n");
    activate_layer(l1, input);

    printf("\nOutput del layer:\n");
    for (int i = 0; i < n_neurons; i++) {
        printf("Neurone %d: %f\n", i, l1->output[i]);
    }

    destroy_layer(l1);

    printf("\nLayer distrutto correttamente.\n");

    return 0;
}