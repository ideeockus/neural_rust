//! This is my try to create artificial neural network from scratch
//!
//! So it may have many disadvanages or mistakes

use rand::prelude::*;


/// Neuron have inputs and weights, after activate() it will calc output value
///
/// Neurons connect to each other between relative layers
struct Neuron {
  input_values: Vec<f64>,
  weights: Vec<f64>,
  bias: f64,
  output_value: f64,
}

/// Abstraction over neuron layer, so I can operate
/// not with matrices and vectors, but with Layer
struct Layer {
  neurons: Vec<Neuron>,
}

/// Provides functions to create, train and compute model
struct Model {
  layers: Vec<Layer>,
}

impl Neuron {
  /// Create a new neuron
  fn new() -> Neuron {
    Neuron {
        input_values: Vec::new(),
        weights: Vec::new(),
        output_value: 0.0,
        bias: rand::random::<f64>(),
      }
  }

  /// RELU function of neuron
  fn relu(x: f64) -> f64 {
    match x {
      x if x < 0.0 => 0.0,
      x if x > 1.0 => 1.0,
      _ => x,
    }
  }

  /// Neuron activation function. Returns neurons output_value
  fn activate(&mut self, use_relu: bool) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..self.weights.len() {
      sum += self.input_values[i] * self.weights[i];
    }
    if !use_relu { return sum }
    self.output_value = Neuron::relu(sum);
    self.output_value
  }
}

impl Layer {
  /// Create a new layer
  /// `neurons_count` - count of neurons in layer
  fn new(neurons_count: usize) -> Layer {
    let mut neurons = Vec::new();
    for i in 0..neurons_count {
      let neuron = Neuron::new();
      neurons.push(neuron);
    }

    Layer { neurons: neurons }

  }

  /// Collect outputs of all neurons in layer to one Vec<f64>
  fn get_neurons_outputs(&self) -> Vec<f64> {
    let mut neurons_outputs: Vec<f64> = Vec::new();
    for neuron in self.neurons.iter() {
      neurons_outputs.push(neuron.output_value);
    }
    neurons_outputs
  }
}

impl Model {
  /// create new model
  /// `inputs_count` - number of inputs
  fn new(inputs_count: usize) -> Model {
    let mut model = Model { layers: Vec::new() };
    model.addLayer(inputs_count);
    model
  }

  /// Add a new layer of neurons
  ///
  fn addLayer(&mut self, neurons_count: usize) {
    self.layers.push(Layer::new(neurons_count));
  }

  /// Initialaze neurons weights
  ///
  /// This weights will correct when network learning
  fn init_weights(&mut self) { // add seed param ?
    // init weights for input neurons
    let input_layer_neurons_count = self.layers.first().unwrap().neurons.len();
    for input_neuron in self.layers.first_mut().unwrap().neurons.iter_mut() {
      input_neuron.weights = vec!(my_random(0.0, 0.3));
    }

    // init weights for other neurons
    for layer_index in 1..self.layers.len() {
      let last_layer_neurons_count = self.layers[layer_index-1].neurons.len();
      for neuron in self.layers[layer_index].neurons.iter_mut() {
        let mut uninitialized_connections_count = last_layer_neurons_count;
        while uninitialized_connections_count > 0 {
          neuron.weights.push(my_random(0.0, 0.2));
          uninitialized_connections_count -= 1;
        }
      }
    }
  }

  /// This is Feed Forward function of network (forward propagation)
  fn ff(&mut self) { //feed forward
    for layer_index in 0..self.layers.len()-1 {
      for current_layer_neuron in self.layers[layer_index].neurons.iter_mut() {
        current_layer_neuron.activate(true);
      }
      // if layer_index == self.layers.len()-1 { break; }

      let current_layer_neuron_outputs = self.layers[layer_index].get_neurons_outputs();
      for next_layer_neuron in self.layers[layer_index+1].neurons.iter_mut() {
        next_layer_neuron.input_values = current_layer_neuron_outputs.clone();
      }
    }
    for output_neuron in self.layers.last_mut().unwrap().neurons.iter_mut() {
      output_neuron.output_value = output_neuron.activate(false);
    }
  }

  /// Train network on data
  fn train(&self, data: Vec<Vec<i32>>) {

    // self.inputs

  }

  /// Put inputs in input neurons and compute final output value
  fn evaluate(&mut self, vac: Vec<f64>) -> Vec<f64> {
    assert_eq!(self.layers[0].neurons.len(), vac.len());
    for index in 0..self.layers[0].neurons.len() {
      let input_neuron = &mut self.layers[0].neurons[index];
      input_neuron.input_values = vec!(vac[index]);
      input_neuron.weights = vec!(1.0);
    }
    self.ff();
    let model_output = self.layers.last().unwrap().get_neurons_outputs();
    model_output
  }
}




impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      let mut model_representation = String::from("Model:\n");
      for layer in self.layers.iter() {
        let mut layer_representation = String::new();
        for neuron in layer.neurons.iter() {
          model_representation.push_str(&neuron.to_string());
        }
        model_representation.push_str("\n\n\n");
      }
      write!(f, "{}", model_representation)
    }
}

impl std::fmt::Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      let format_floats = |floats_vec: &Vec<f64>| {
        floats_vec.iter().map(|x| (x * 1000.0).round() / 1000.0).collect()
      };
      // let weights = 
      let weights: Vec<f64> = format_floats(&self.weights);
      let input_values: Vec<f64> = format_floats(&self.input_values);
      let neuron_display = format!("{side}\nweights {:?}\n\
        inputs {:?}\noutputs {:^7.5}\n{side}\n",
       weights, input_values, self.output_value, side="+++++++++ +++++++++");
      write!(f, "{}", neuron_display)
    }
}


/// I use it to get random values in specified range
fn my_random(min: f64, max: f64) -> f64{
  rand::random::<f64>()*(max - min) + min
}

/// This function is not part of neural network,
/// it just make data for training
fn make_train_data(data_pairs_count: i32) -> Vec<Vec<f64>> {
  let mut train_data: Vec<Vec<f64>> = Vec::new();
  for i in 0..data_pairs_count {
    let a1 = my_random(-100.0, 100.0);
    let a2 = my_random(-100.0, 100.0);
    let sum = a1 + a2;
    let train_pair = vec![a1, a2, sum];
    train_data.push(train_pair);
  }
  train_data
}


/// This is start point
fn main() {
    let mut model = Model::new(2);
    model.addLayer(5);
    model.addLayer(1);
    model.init_weights();

    // println!("{}", model.layers[1].neurons[1]);
    // println!("{}", model);

    // let train_data = make_train_data(50);
    // model.train(train_data);

    let results = model.evaluate(vec!(2 as f64, 2 as f64));
    println!("{}", model);
    println!("{:?}", results);
}




#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }

  #[test]
  fn activation_func_test() {
    let mut neuron1 = super::Neuron::new();
    neuron1.input_values = vec![0.5, 0.2, 0.4];
    neuron1.weights = vec![0.2, -0.1, 0.3];
    assert_eq!(neuron1.activate(true), 0.2);

    let mut neuron2 = super::Neuron::new();
    neuron2.input_values = vec![0.8, 0.6];
    neuron2.weights = vec![2.0, -5.0];
    assert_eq!(neuron2.activate(true), 0.0);
    assert_eq!(neuron2.activate(false), -1.4);
  }

  #[test]
  fn relu_test() {
    assert_eq!(super::Neuron::relu(-8.0), 0.0);
    assert_eq!(super::Neuron::relu(0.5), 0.5);
    assert_eq!(super::Neuron::relu(1.0), 1.0);
    assert_eq!(super::Neuron::relu(20.0), 1.0);
  }

  #[test]
  fn model_test() {
    let test_model = super::Model::new(2);

  }

}