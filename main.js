const { train, predict, unfunn } = require("./nn/nn");

// let nn = [[[0, 1], [0, 1]], [[1, 1]]];
let nn = unfunn(2, [2, 1]);
let test = [
  [[0, 1], [1]],
  [[1, 0], [0]],
  [[2, 1], [1]],
  [[2, 0], [0]],
  [[3, 1], [1]],
  [[3, 0], [0]],
  [[4, 1], [1]],
  [[4, 0], [0]],
  [[5, 1], [1]],
  [[5, 0], [0]],
  [[6, 1], [1]],
  [[6, 0], [0]]
];

let trained = nn;

for (let epo = 0; epo < 1; epo++) {
  if (!(epo % 10)) {
    console.log(`Epoch: ${epo}`);
    console.log(predict(trained, [0, 1]));
    console.log(predict(trained, [1, 0]));
    console.log();
    console.log(trained);
    console.log("\n\n\n");
  }

  trained = train(trained, test, 1);
}
