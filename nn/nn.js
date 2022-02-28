const alpha = 0.5; // Just some constant alpha, changing this does stuff
const maxEpochs = 1000; // The most training cycles that can be done in one shot

// Value: the Number to run through the function
// Returns a Number
function sigmoid(value) {
  try {
    return 1 / (1 + Math.exp(-value));
  } catch (err) {
    return 0;
  }
}

// Layer: an array of arrays of Numbers, where each subarray is a node
// Input: an array of Numbers
// Returns an array of the dot product of every node run through the sigmoid fn
function activateLayer(layer, input) {
  return layer.map(node =>
    sigmoid(
      node
        .map((weight, i) => weight * input[i]) // node[i] * input[i]
        .reduce((prev, curr) => prev + curr) // Sum for dot product
    )
  );
}

// Layers: an array of arrays of arrays of weights
//         [layer1, layer2, ..., layer_n], where each layer is an array of nodes
//         [node1, node2, ..., node_m], where each node is an array of weights
//         [weight1, weight2, ..., weight_i], where each weight is a Number
//  Input: an array of size n, where n is the size of a node in the first layer
// Returns the activation result of the final layer
function unfunn(layers, input) {
  return layers.length
    ? unfunn(layers.slice(1), activateLayer(layers[0], input))
    : input;
}

// NeuralNetwork: the neural net we want to train
//  DeltaNetwork: delta values with the same structure as NeuralNetwork
//                [layer1Deltas, layer2Deltas, ..., layer_nDeltas]
//                [node1Delta, node2Delta, ..., node_mDelta]
function update(neuralNetwork, deltaNetwork) {
  // These are in reverse order, so we need to flip them
  neuralNetwork.reverse();
  deltaNetwork.reverse();

  neuralNetwork.map((layer, i) =>
    layer.map((node, j) =>
      node.map((weight, k) => {
        return weight + deltaNetwork[i][j] * (alpha + alpha * alpha);
      })
    )
  );

  return neuralNetwork.map((layer, i) =>
    layer.map((node, j) =>
      node.map(weight => weight + deltaNetwork[i][j] * (alpha + alpha * alpha))
    )
  );
}

//       NN: the neural network we want to train
//   Result: the result of our neural network
// Expected: the actual answer
function backprop(nn, result, expected) {
  nn.reverse(); // Output layer first

  let deltaNetwork = [];

  nn.forEach((layer, i) => {
    // Not output layer
    if (i) {
      deltaNetwork.push(
        layer.map(node =>
          nn[i - 1]
            .map((nextNode, j) =>
              nextNode
                .map(weight => weight * deltaNetwork[i - 1][j])
                .reduce((prev, curr) => prev + curr)
            )
            .reduce((prev, curr) => prev + curr)
        )
      );
    }

    // Output layer
    else {
      deltaNetwork.push(
        layer.map((node, j) => {
          return result[j] * (1 - result[j]) * (expected[j] - result[j]);
        })
      );
    }
  });

  return update(nn, deltaNetwork);
}

//      NN: an unnecessarily functional neural network
//  Epochs: the number of loops to perform
//    Data: the test data set; each entry is in the form [input, output]
function train(nn, data, epochs) {
  if (epochs > maxEpochs) epochs = maxEpochs;
  if (epochs < 1) return nn; // Base case

  // Update the weights in the NN for every training case
  data.forEach(([input, output]) => {
    nn = backprop(nn, unfunn(nn, input), output);
  });

  return train(nn, data, --epochs); // Loop
}

let nn = [[[0, 1], [1, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0, 0]]];
let test = [[[3, 5], [1]]];

// trained network
trained = train(nn, test, 10000);

// result from training
// console.log(trained);

console.log(unfunn(trained, test[0][0]));
