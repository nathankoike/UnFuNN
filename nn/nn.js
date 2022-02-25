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
//         [layer1, layer2, ..., layern], where each layer is an array of nodes
//         [node1, node2, ..., nodem], where each node is an array of weights
//         [weight1, weight2, ..., weighti], where each weight is a Number
//  Input: an array of size n, where n is the size of a node in the first layer
// Returns the activation result of the final layer
function unfunn(layers, input) {
  return layers.length
    ? unfunn(layers.slice(1), activateLayer(layers[0], input))
    : input;
}

console.log(unfunn([[[0.5, 0.5]]], [3, 5]));
