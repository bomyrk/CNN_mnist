# Fashion-MNIST CNN Classifier

A Keras/TensorFlow convolutional neural network for classifying Fashion-MNIST apparel images.

## What I Built

I developed a CNN architecture that achieves **91.44% test accuracy** on the Fashion-MNIST dataset. The model features:

- **Architecture**: 3 convolutional blocks (32→64→128 filters) with MaxPooling2D, Dropout regularization (0.2-0.25), and a dense classifier (512 units)
- **Training**: 5-fold cross-validation with SGD optimizer (lr=0.01, momentum=0.9), ModelCheckpoint for best weights, and EarlyStopping
- **Monitoring**: TensorBoard integration for real-time loss/accuracy visualization
- **Persistence**: Model saved as `.h5` format with best weights from validation

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) - 70,000 grayscale images (28×28 pixels) across 10 apparel classes:

- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Results

- **Test Accuracy**: 91.44%
- **Cross-validation Mean**: 91.94% ± 0.32%
- **Training**: 10 epochs per fold with batch size 32

## How to Run

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook fashion_mnist_cnn.ipynb
```

### Docker

#### Prerequisites
First, train and export the model:
```bash
pip install -r requirements.txt
python train_model.py
```

This creates `models/on_the_go/fashion_mnist_api_final.keras` required for serving.

#### Build and Run
```bash
# Build and start all services
docker-compose up --build

# Scale web containers (e.g., 3 replicas)
docker-compose up --scale web=3

# Scale down
docker-compose up --scale web=1
```

#### Test Endpoints
```bash
# Health check
curl http://localhost/health

# Predict with image upload
curl -X POST http://localhost/predict -F "image=@test_image.png"

# Predict with pixel array (JSON)
curl -X POST http://localhost/predict_array \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,0,...]}'

# View API documentation
open http://localhost/docs
```

#### TensorBoard
Access training visualizations at http://localhost:6006

## Tech Stack

- TensorFlow 2.x / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (cross-validation)
- FastAPI (serving API)
- Docker & Docker Compose
- Nginx (load balancer)
