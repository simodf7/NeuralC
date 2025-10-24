#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "activation_function.h"


int main(void) {

    srand((unsigned int)time(NULL)); // inizializza generatore random

    printf("=== Creazione rete neurale di test ===\n");

    // Crea una rete con 2 layer
    struct network* net = create_network(3,2);

    // Crea i layer
    struct dense_layer* l1 = create_dense_layer(3, 4, reLU); // 3 neuroni, 4 input
    struct dense_layer* l2 = create_dense_layer(2, 3, reLU); // 2 neuroni, input dai 3 output precedenti
    struct softmax_layer* l3 = create_softmax_layer(2); 

    // Aggiungi i layer alla rete
    add_layer(net, (struct layer*) l1, 0);
    add_layer(net, (struct layer*) l2, 1);
    add_layer(net, (struct layer*) l3, 2); 

    // Inizializza i pesi
    initialize_network(net);

    // Definisci un input (4 valori)
    float input[4] = {1.0, -2.0, 0.5, 3.0};

    // Esegui il forward pass
    forward(net, input);

    // Stampa l’output finale
    printf("\n Output Rete: ");
    for(int i=0; i<2; i++){
        printf("%f ", net->output[i]);
    }

    // Libera memoria
    destroy_network(net);

    printf("\n=== Esecuzione completata ===\n");
    return 0;

}
