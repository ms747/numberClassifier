let element;
let context;
let imageFromCanvas;
let resetButton;
let trainButton;
const model = tf.sequential();
const out = tf.tensor2d([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]);


function setup() {
  let x = createCanvas(28, 28);
  element = x.canvas;
  context = element.getContext("2d");
  background(0);
  resetButton = createButton("Reset");
  trainButton = createButton("Train");
  resetButton.position(0, 100);
  trainButton.position(50, 100);
  resetButton.mousePressed(resetFunction);
  trainButton.mousePressed(trainFunction);

  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 3],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "VarianceScaling"
    })
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "VarianceScaling"
    })
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 10,
      kernelInitializer: "VarianceScaling",
      activation: "softmax"
    })
  );

  const LEARNING_RATE = 0.15;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
}

function resetFunction() {
  imageFromCanvas = tf.reshape(tf.fromPixels(element),[1,28,28,3]);
  imageFromCanvas.dtype = "float32";
  imageFromCanvas.div(tf.scalar(255));
  model.predict(imageFromCanvas).print()
  background(0)
}

function trainFunction() {
  imageFromCanvas = tf.reshape(tf.fromPixels(element),[1,28,28,3]);
  imageFromCanvas.dtype = "float32";
  imageFromCanvas.div(tf.scalar(255)); 
  model.fit(imageFromCanvas,out,{epochs:1}).then((his)=>{
    console.log(his.history.loss);
    background(0);
  })

} 

function draw() {}

function mouseDragged() {
  stroke(255);
  strokeWeight(2);
  line(mouseX, mouseY, pmouseX, pmouseY);
}
