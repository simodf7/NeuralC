struct neuron{
    int n_inputs; // aggiunto per rendere dinamico il numero di input di un neurone 
    float* weights; 
    float bias; 
    float (*activation_function)(float);  // puntatore a funzione che prende float e restituisce float
    float output; 
    float local_gradient; // da capire
};


struct neuron* create_neuron(float (*func)(float), int n_inputs); 
void assign_weights(struct neuron* n, float* w); 
void activate_neuron(struct neuron* n, float* input); 
float* initial_weights(int n_inputs); 
void destroy_neuron(struct neuron* n); 