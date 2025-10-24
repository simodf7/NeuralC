#ifndef NEURON_H
#define NEURON_H

struct neuron{
    int n_inputs; // aggiunto per rendere dinamico il numero di input di un neurone 
    float* weights; 
    float bias; 
    float output; 
    float local_gradient; // da capire 
    // tolta la funzione di attivazione, non ha senso perche è meglio inserirla nel layer
};


struct neuron* create_neuron(int n_inputs); 
void assign_weights(struct neuron* n, float* w); 
void activate_neuron(struct neuron* n, float* input); 
float* initial_weights(int n_inputs); 
void destroy_neuron(struct neuron* n); 
#endif 