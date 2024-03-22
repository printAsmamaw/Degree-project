
# Apple Disease Identification Using Computer Vision

The Apple Disease Identification Project is a digital tool designed to assist farmers, orchard managers, and agricultural specialists in identifying diseases affecting apple trees. Leveraging technologies such as image recognition and machine learning, the project aims to streamline the process of diagnosing apple tree diseases. Users can upload images of affected leaves, fruits, or the entire tree, and the system provides real-time analysis, suggesting potential diseases and treatment options. By enabling rapid and accurate identification, the project helps farmers make informed decisions to protect their orchards and optimize crop yields.


## Authors

- [@printAsmamaw](https://github.com/printAsmamaw)


## Deployment

To deploy this machine learning model on mobile application we convert the .h5 model to .tflite model

```bash
  import tensorflow as tf
   model = tf.keras.models.load_model('your_model.h5')
```
```bash
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
```
```bash
  with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

After that we deploye the tflite model on flutter mobile application by using
```bash
  String res = await Tflite.loadModel(
  model: "assets/appe_disease_model.tflite",
  labels: "assets/labels.txt",
  numThreads: 1, 
  isAsset: true, 
  useGpuDelegate: false 
);
```


## 🛠 Skills
Python,Tensorflow,Pandas,Numpy,Conventional Neural Network(CNN),flutter,


## Installation

Installation for our project with pip

```bash
  !pip install tensorflow
  !pip install numpy
  !pip install pandas
  cd apple project
```
    
## Demo

Insert gif or link to demo


## Roadmap

- Dataset collection for healthy and disease apple leaves

- Preprocess the image dataset

- Image augmentation

- Train test split

- Model selection Conventional Neural Network

- Add the parameter on Conventional Neural Network based on the collected dataset

- Model training with agusted epoch

- Model evaluation with training and testing dataset

- Mobile application development

- Model deployment



## Usage/Examples

```python
model =Sequential([ InputLayer(input_shape=input_shape),
Conv2D(32, (3,3),padding='same', activation='relu'), MaxPooling2D((2, 2)),

Conv2D(64, (3,3),padding='same', activation='relu'), MaxPooling2D((2, 2)),

Conv2D(128, (3,3),padding='same', activation='relu'), MaxPooling2D((2, 2)),

Conv2D(256, (3,3), padding='same',activation='relu'),
MaxPooling2D((2, 2)),

Conv2D(512, (3, 3),padding='same',activation='relu'),
MaxPooling2D((2, 2)),

Conv2D(1024, (3, 3),padding='same',activation='relu'),
MaxPooling2D((2, 2)), Dropout(0.5),

Conv2D(1024, (3, 3),padding='same',activation='relu'),
MaxPooling2D((2, 2)), Dropout(0.5),

Flatten(),

Dense(1024,activation='relu'),
Dense(128,activation='relu'),
Dense(64,activation='relu'),

Dense(5,activation='softmax')
                  ])
```


## Screenshots
# Sample Dataset
![Healthy and Block Ror](https://github.com/printAsmamaw/Degree-project/assets/122156542/ca75d748-1b0a-4b6c-84a0-4be76c93b31c)

![Mosaic and Block Rot](https://github.com/printAsmamaw/Degree-project/assets/122156542/0cced243-0382-41cb-b2d6-0b0853c73018)

# Evaluation of the model

![Model Evaluation](https://github.com/printAsmamaw/Degree-project/assets/122156542/26ea7319-27a1-42ae-9960-cdcaa2af17ec)

