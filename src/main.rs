use rand::prelude::*;


struct Neuron {
  input_values: Vec<f64>,
  weights: Vec<f64>,
  bias: f64,
  output_neurons: Vec<Neuron>,
}

struct Layer {
  neurons: Vec<Neuron>,
}

struct Model {
  layers: Vec<Layer>,
  inputs: Vec<Neuron>,
}

impl Neuron {
  fn relu(x: f64) -> f64 {
    match x {
      x if x < 0.0 => 0.0,
      x if x > 1.0 => 1.0,
      _ => x,
    }
  }

  fn activation_func(input_values: Vec<f64>, weights: Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..weights.len() {
      sum += input_values[i] * weights[i];
    }

    Neuron::relu(sum)
  }

  fn connect_to(&mut self, neuron: Neuron) {
    self.output_neurons.push(neuron);
  }
}

impl Layer {
  // add code here
  fn new(neurons_count: i32) -> Layer {
    let mut neurons = Vec::new();
    for i in 0..neurons_count {
      let neuron = Neuron {
        input_values: Vec::new(),
        weights: Vec::new(),
        output_neurons: Vec::new(),
        bias: rand::random::<f64>(),
      };
      neurons.push(neuron);
    }


    Layer { neurons: neurons }

  }
}

impl Model {
  // add code here
  fn new(inputs_count: usize) -> Model {
    Model { layers: Vec::new(), inputs: Vec::with_capacity(inputs_count) }
  }

  fn addLayer(&mut self, neurons_count: i32) {
    // if self.layers.len() == 0 {
    //   self.layers.push(Layer::new(neurons_count));
    //   for input_neuron in 
    //   for neuron in self.layers[0].neurons {

    //   }
    // }

    let mut layers = self.layers;
    let layers_count = layers.len();
    layers.push(Layer::new(neurons_count));

    for out_neuron in layers[layers_count - 2].neurons.iter_mut() {
      for input_neuron in layers[layers_count - 1].neurons.into_iter() {
        out_neuron.connect_to(input_neuron);
      }
    }
    
  }

  fn train(&self, data: Vec<Vec<i32>>) {
    // self.inputs

  }

  fn evaluate(&self, vac: Vec<i32>) {

  }
}

fn my_random(min: i32, max: i32) -> i32{
  rand::random::<i32>()*(max - min) + min
}

fn make_train_data(data_pairs_count: i32) -> Vec<Vec<i32>> {
  let mut train_data: Vec<Vec<i32>> = Vec::new();
  for i in 0..data_pairs_count {
    let a1 = my_random(-100, 100); //*(max-min)+min
    let a2 = my_random(-100, 100);
    let sum = a1 + a2;
    let train_pair = vec![a1, a2, sum];
    train_data.push(train_pair);
  }

  train_data
}

fn main() {
    let mut model = Model::new(2);
    // let layer1 = Layer::new(5);
    model.addLayer(5);
    model.addLayer(1);

    let train_data = make_train_data(50);
    model.train(train_data);

    let results = model.evaluate(vec!(2, 2));
}


// fn activation_func(input_values: Vec<f64>, weights: Vec<f64>) -> f64 {
//   let mut sum: f64 = 0.0;
//   for i in 0..weights.len() {
//     sum += input_values[i] * weights[i];
//   }

//   relu(sum)
// }

// fn relu(x: f64) -> f64 {
//   match x {
//     x if x < 0.0 => 0.0,
//     x if x > 1.0 => 1.0,
//     _ => x,
//   }
// }




#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }

  #[test]
  fn activation_func_test() {
    let input_values = vec![0.5, 0.2, 0.4];
    let weights = vec![0.2, -0.1, 0.3];
    assert_eq!(super::Neuron::activation_func(input_values, weights), 0.2);

    let input_values = vec![0.8, 0.6];
    let weights = vec![2.0, -5.0];
    assert_eq!(super::Neuron::activation_func(input_values, weights), 0.0);
    // noop!();
  }

  #[test]
  fn relu_test() {
    assert_eq!(super::Neuron::relu(-8.0), 0.0);
    assert_eq!(super::Neuron::relu(0.5), 0.5);
    assert_eq!(super::Neuron::relu(1.0), 1.0);
    assert_eq!(super::Neuron::relu(20.0), 1.0);
  }

}