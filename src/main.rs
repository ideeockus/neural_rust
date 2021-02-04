//! This is my try to create artificial neural network from scratch
//!
//! So it may have many disadvanages or mistakes

use std::io::Write;
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

  /// RELU function of neuron (scaling function)
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
      sum += self.input_values[i] * self.weights[i] + self.bias;
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

  /// Collect inputs of all neurons in layer to one Vec<f64>
  fn get_neurons_inputs(&self) -> Vec<f64> {
    let mut neurons_inputs: Vec<f64> = Vec::new();
    for neuron in self.neurons.iter() {
      neurons_inputs.extend(&neuron.input_values);
    }
    neurons_inputs
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
      input_neuron.weights = vec!(my_random(0.0, 0.08));
      input_neuron.bias = my_random(0.0, 0.2);
    }

    // init weights for other neurons
    for layer_index in 1..self.layers.len() {
      let last_layer_neurons_count = self.layers[layer_index-1].neurons.len();
      for neuron in self.layers[layer_index].neurons.iter_mut() {
        neuron.bias = my_random(0.0, 0.2);
        let mut uninitialized_connections_count = last_layer_neurons_count;
        while uninitialized_connections_count > 0 {
          neuron.weights.push(my_random(0.0, 0.08));
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

  // fn back_propagation(&mut self, data_output: Vec<f64>) {
  //   let correct_output = data_output;
    
  //   for layer_index in (1..self.layers.len()).rev() {
  //     let previous_layer_index = layer_index-1;
  //     let previous_layer_len = self.layers[previous_layer_index].neurons.len();
  //     let mut weight_corrections: Vec<f64> = vec![0.0; previous_layer_len];
  //     let mut bias_corrections: Vec<f64> = vec![0.0; previous_layer_len];
  //     // let use_relu: bool = if layer_index == self.layers.len() { false } else { true };

  //     for neuron_index in 0..self.layers[layer_index].neurons.len() {
  //       let neuron = &mut self.layers[layer_index].neurons[neuron_index];
  //       let activation_value = neuron.output_value;
  //       let previous_layer_activations = self.layers[previous_layer_index].get_neurons_outputs();
  //       for i in 0..previous_layer_len {
  //         let weight_correction = 2.0 * (activation_value - correct_output[neuron_index])
  //                                   * activation_value * previous_layer_activations[i];
  //         // weight_corrections.push(weight_correction);
  //         weight_corrections[i] += weight_correction;
  //         let bias_correction = 2.0 * (activation_value * correct_output[neuron_index])
  //                                   * activation_value;
  //         bias_corrections[i] += bias_correction;
  //       }
  //     }
  //     for previous_layer_neuron_index in 0..previous_layer_len {
  //       let previous_layer_neuron = &mut self.layers[previous_layer_index]
  //                                           .neurons[previous_layer_neuron_index];
  //       previous_layer_neuron.
  //     }
  //   }

  // fn back_propagation(&mut self, data_output: Vec<f64>) {
  //   let mut correct_output = data_output;

  //   for layer_index in (1..self.layers.len()).rev() {
  //     let mut previous_layer_activations = self.layers[layer_index-1].get_neurons_outputs();
  //     for neuron_index in 0..self.layers[layer_index].neurons.len() {
  //       let neuron = &mut self.layers[layer_index].neurons[neuron_index];
  //       let activation_value = neuron.output_value;
  //       // let mut previous_layer_output_corrections: Vec<f64> = previous_layer_activations.clone();
  //       let bias_correction = 2.0 * (activation_value - correct_output[neuron_index])
  //                                   * activation_value;
  //       neuron.bias -= bias_correction;
  //       for i in 0..neuron.input_values.len() {
  //         let weight_correction = 2.0 * (activation_value - correct_output[neuron_index])
  //                                   * activation_value * previous_layer_activations[i];
  //         neuron.weights[i] -= weight_correction;

  //         let previous_layer_output_correction = 2.0 
  //         * (activation_value - correct_output[neuron_index]);
  //         previous_layer_activations[i] -= previous_layer_output_correction;
  //       }
        
  //     }
  //     correct_output = previous_layer_activations;
  //   }
  // } 

  fn back_propagation(&mut self, data_output: Vec<f64>, descent_speed: f64) {
    // println!("BACK PROPOGATION");
    let mut correct_output = data_output;

    for layer_index in (0..self.layers.len()).rev() {
      // println!("{:?} слой", layer_index);
      let mut previous_layer_activations = if layer_index==0 {
        self.layers[layer_index].get_neurons_inputs()
      } else {
        self.layers[layer_index-1].get_neurons_outputs()
      };
      // println!("активации предыдущего слоя: {:?}", previous_layer_activations);
      // println!("correct_output: {:?}", correct_output);
      let use_relu = if layer_index==self.layers.len()-1 {false} else {true};
      // println!("use_relu {:?}", use_relu);
      for neuron_index in 0..self.layers[layer_index].neurons.len() {
        // println!("{:?} нейрон", neuron_index);
        let neuron = &mut self.layers[layer_index].neurons[neuron_index];
        let activation_value = neuron.activate(use_relu);
        // println!("его активация равна {:?}", activation_value);
        let bias_correction = (activation_value - correct_output[neuron_index]) * activation_value;
        neuron.bias -= bias_correction * descent_speed;
        // println!("коррекция bias {:?}", bias_correction);
 
        for i in 0..neuron.input_values.len() {
          let weight_correction = (activation_value - correct_output[neuron_index])
                      * activation_value * previous_layer_activations[i];
          // println!("коррекция weight[{}] {:?}", i, weight_correction);

          let previous_layer_output_correction = (activation_value - correct_output[neuron_index]) * neuron.weights[i];
          // println!("{:?}", );
          neuron.weights[i] -= weight_correction * descent_speed;
          previous_layer_activations[i] -= previous_layer_output_correction * descent_speed;
        }
      }
      correct_output = previous_layer_activations.into_iter().map(|o| Neuron::relu(o)).collect();
      // println!("коррекции активаций предыдущего слоя {:?}", correct_output);
      // println!();
    }
  } 

  /// Train network on data
  fn train(&mut self, data: (Vec<f64>, Vec<f64>)) {
    let train_data_input = data.0;
    let train_data_output = data.1;

    self.evaluate(train_data_input);
    self.back_propagation(train_data_output, 0.05);
    // self.inputs

  }

  /// Put inputs in input neurons and compute final output value
  fn evaluate(&mut self, vac: Vec<f64>) -> Vec<f64> {
    assert_eq!(self.layers[0].neurons.len(), vac.len());
    for index in 0..self.layers[0].neurons.len() {
      let input_neuron = &mut self.layers[0].neurons[index];
      input_neuron.input_values = vec!(vac[index]);
      // input_neuron.weights = vec!(1.0);
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
        // let mut layer_representation = String::new();
        let layer_splitter = "========= ========= =========\n";
        model_representation.push_str(layer_splitter);
        for neuron in layer.neurons.iter() {
          model_representation.push_str(&neuron.to_string());
        }
        model_representation.push_str(layer_splitter);
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
      let bias: f64 = (self.bias * 1000.0).round() / 1000.0;
      let input_values: Vec<f64> = format_floats(&self.input_values);
      let neuron_display = format!("{side}\n\
        inputs {:?}\n\
        weights {:?}\n\
        bias {}\n\
        outputs {:^7.5}\n{side}\n",
       input_values, weights, bias, self.output_value, side="+++++++++ +++++++++");
      write!(f, "{}", neuron_display)
    }
}


/// I use it to get random values in specified range
fn my_random(min: f64, max: f64) -> f64{
  rand::random::<f64>()*(max - min) + min
}

/// This function is not part of neural network,
/// it just make data for training
fn make_train_data(data_pairs_count: i32) -> Vec<(Vec<f64>, Vec<f64>)> {
  let mut train_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
  // let mut train_data_input: Vec<f64> = Vec::new();
  // let mut train_data_output: Vec<f64> = Vec::new();
  for i in 0..data_pairs_count {
    let a1 = my_random(0.0, 1.0);
    let a2 = my_random(0.0, 1.0);
    let sum = a1 + a2;
    let train_data_input = vec![a1, a2];
    let train_data_output = vec![sum];
    
    train_data.push((train_data_input, train_data_output));
  }
  train_data
}


/// This is start point
fn main() {
    let mut model = Model::new(2);
    // model.addLayer(5);
    model.addLayer(1);
    model.init_weights();

    // println!("{}", model.layers[1].neurons[1]);
    // println!("{}", model);

    let result1 = model.evaluate(vec!(0.3 as f64, 0.5 as f64));
    println!("{}", model);
    // results.push(result1[0]);

    let train_data = make_train_data(50000);
    // println!("{:?}", train_data);
    for train_pair in train_data.into_iter() {
      model.train(train_pair);
    }
    let mut results: Vec<f64> = Vec::new();

    // model.back_propagation(vec!(4.0), 0.05);
    println!("Модель после обучения");
    let result2 = model.evaluate(vec!(0.3 as f64, 0.5 as f64));
    println!("{}", model);
    results.push(result2[0]);

    // for i in 0..10 {
    //   model.back_propagation(vec!(4.0), 0.05);
    //   let tmp_result = model.evaluate(vec!(2 as f64, 2 as f64));
    //   results.push(tmp_result[0]);
    //   println!("{}", model);
    // }
    // model.back_propagation(vec!(4.0), 0.05);
    // let result3 = model.evaluate(vec!(2 as f64, 2 as f64));
    // println!("{}", model);
    // results.push(result3[0]);

    println!("{:?} -> {:?}", result1, result2);

    loop {
      print!("Входные данные нейросети:   ");
      std::io::stdout().flush().unwrap();
      let mut input_line = String::new();
      std::io::stdin().read_line(&mut input_line).unwrap();

      let user_inputs = input_line.split(" ").map(|x| x.trim().parse::<f64>().unwrap()).collect::<Vec<f64>>();
      println!("Входные данные: {:?}", user_inputs);
      let res = model.evaluate(user_inputs);

      println!("{:?}", res);
    }
    // println!("");
    // println!("{:?}", results);
    // println!("{:?} -> {:?}", result1, result3);

    // results.into_iter().map(
    //   |r| print!("{}", r)
    //   );

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