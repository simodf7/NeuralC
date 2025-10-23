struct layer{
    int n_neurons; 
    int n_inputs_per_neuron; 
    struct neuron** neurons;  // array di puntatori a neuroni
    float (*activation_function)(float); 
    float* output;  // output di ogni neurone
};



struct layer* create_layer(int n_neurons, int n_inputs_per_neuron, float (*func)(float)); 
void initialize_weights(struct layer* l); 
void activate_layer(struct layer* l, float* input); 
void destroy_layer(struct layer* l); 
