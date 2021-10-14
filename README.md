# Model Deployment/Serving with FastAPI

The project is built with FastAPI that exposes a REST API endpoint, allowing applications to retrieve a prediction result for a deep neural network/multi-layer perceptron model developed during my time as an undergraduate (Optimisation and Prediction of Bus Arrival Timings).

# Getting Started #

## Dependencies
1. Docker Engine
2. Kubernetes/Minikube (optional)

## Installation

## Building the Model Training Docker container
```
cd pipeline/
sh build.sh
```

### Building the Prediction Service Docker container

```
cd prediction-service/
sh build.sh
```

## Running the program

## Running the Model Training via Docker container
```
cd pipeline/

```

### Running the Prediction Service Docker container
```
cd prediction-service/
sh run.sh
```

### Accessing OpenAPI UI
1. Make sure that the prediction service container is running (previous step)
2. Access UI at http://0.0.0.0:80/docs
3. Click on "Try it out" for the /predict endpoint and test with the following sample input

```
#Concatenated bus stop code (46529 and 46491)
bus_stop_code = 4652946491

#1 to 7 (Mon - Sun)
day = 4 

#Aggregated and embedded by 10 minute timeframes from 0510 to 1240
time = 100
```

## Deployment

### Deploying Prediction Service in Kubernetes/Minikube

```
sh deploy.sh
```