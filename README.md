# Federated Learning with Edge Intelligence in Smart Healthcare

## 📌 Project Overview
This project explores the integration of **Federated Learning (FL)** and **Edge Intelligence** to enhance data refinement and improve the accuracy of global models in smart healthcare systems. By training models locally on edge devices, we ensure data privacy and leverage distributed data sources to create robust global models applicable to various healthcare challenges.

## 🛠 Tech Stack & Tools
- **Programming Language:** Python
- **Frameworks & Libraries:** TensorFlow, PyTorch
- **Techniques Implemented:**
  - Federated Averaging (**FedAvg**)
  - Federated Stochastic Gradient Descent (**FedSGD**)
- **Dataset:** MHEALTH (Mobile Health) Dataset

## 📊 Dataset Details
The **MHEALTH (Mobile Health) Dataset** comprises body motion and vital signs recordings from ten volunteers of diverse profiles performing various physical activities. Sensors placed on the chest, right wrist, and left ankle measured motion (acceleration, rate of turn, magnetic field orientation) and 2-lead ECG measurements. This dataset is instrumental in human behavior analysis based on multimodal body sensing.

## 🚀 Features & Methodology
- **Data Preprocessing:**
  - Standardized and normalized sensor data.
  - Addressed class imbalances to enhance model accuracy.
- **Local Model Training:**
  - Implemented algorithms to train models on edge devices, ensuring data remains local to preserve privacy.
- **Federated Learning Techniques:**
  - Integrated **Federated Averaging (FedAvg)** to aggregate local models into a global model.
  - Applied **Federated Stochastic Gradient Descent (FedSGD)** for model updates.
- **Edge Intelligence:**
  - Leveraged edge computing resources to process data closer to the source, reducing latency and bandwidth usage.

## ⚙️ Implementation Steps
1. **Data Acquisition:**
   - Downloaded the MHEALTH dataset from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset).
2. **Data Preprocessing:**
   - Loaded and cleaned the dataset.
   - Standardized features and addressed class imbalances.
3. **Local Model Training:**
   - Distributed data across simulated edge devices.
   - Trained local models using TensorFlow.
4. **Federated Aggregation:**
   - Applied FedAvg and FedSGD to combine local models into a global model.
5. **Evaluation:**
   - Assessed model performance using metrics like accuracy, precision, and recall.

## 📈 Performance Benchmarks
- **Federated Averaging (FedAvg):** Achieved a global model accuracy of **93%**.
- **Federated Stochastic Gradient Descent (FedSGD):** Achieved a global model accuracy of **95%**.

## 🔍 Model Output
The global model provides:
- **Predictions:** Classifies physical activities based on sensor data.
- **Performance Metrics:** Accuracy, precision, recall, and F1-score.
- **Visualizations:** Training and validation loss curves, confusion matrices.

## ⚙️ Setup & Usage
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/Federated-Learning-Healthcare.git](https://github.com/Tajumulla/Federated-Learning-Healthcare)
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd Federated-Learning-Healthcare
   ```

3. **Install Dependencies:** Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Training Script:**
   ```bash
   python train.py
   ```

5. **Evaluate the Model:**
   ```bash
   python evaluate.py
   ```

## 📚 References
- MHEALTH Dataset - UCI Machine Learning Repository
- Federated Learning - A Comprehensive Overview

## 🤝 Contributions
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.
