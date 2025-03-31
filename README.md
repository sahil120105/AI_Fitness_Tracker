# 🏋️ AI Fitness Tracker

## 📌 Project Overview
The **AI Fitness Tracker** is designed to accurately predict the type of exercise a user is performing and count the number of repetitions. The future scope includes detecting incorrect form during gym exercises, making it a potential AI-powered personal trainer.

## 📊 Data Collection
- The dataset consists of **accelerometer and gyroscope** sensor data collected from **wristbands** during strength training.
- Data was recorded from **5 participants** performing **five fundamental barbell exercises**:
  - **Bench Press**
  - **Deadlift**
  - **Overhead Press**
  - **Row**
  - **Squat**
- Each participant performed:
  - **3 sets of 5 reps** (heavy weight based on 1RM)
  - **3 sets of 10 reps** (medium weight based on 1RM)
- **Resting Data** was collected between sets (standing, walking, sitting).
- Additional improper form data was collected for **bench press** (e.g., bar placement too high, not touching chest).
- **Sensor Data Includes**:
  - Timestamp
  - Accelerometer (x, y, z values)
  - Gyroscope (x, y, z values)
  
## 🔍 Data Preprocessing & Analysis
- **Data Aggregation**: Used a **0.20-second step size** (five instances per second) by taking the **mean** of numerical attributes and **mode** of categorical attributes.
- **Exploratory Data Analysis (EDA)**:
  - Visualized distributions and patterns.
  - Identified trends in sensor readings for different exercises.
- **Outlier Detection**:
  - Applied **IQR method, Chauvenet’s criterion (final choice), and LOF** to remove inaccurate data.
- **Feature Engineering**:
  - **Set Durations**
  - **Butterworth Low-pass Filtering**
  - **Principal Component Analysis (PCA)**
  - **Sum of Squares Attributes (Acceleration & Gyroscope vectors)**
  - **Temporal Abstraction & Frequency Features** (Fourier Transforms)

## 🤖 Machine Learning Pipeline
1. **Feature Selection**: Decision tree analysis to identify the most relevant features.
2. **Model Training**:
   - **Algorithms Used**:
     - 🎯 **Random Forest** (Best performer - **98% accuracy**)
     - 🤖 Neural Network
     - 🌲 Decision Tree
     - 📈 K-Nearest Neighbors (KNN)
     - 📉 Naïve Bayes
   - **Hyperparameter Tuning & k-Fold Cross-Validation** applied for optimization.

## 🔢 Repetition Counting
- **Low-pass filtering** applied for smooth time-series data.
- **argrelextrema Algorithm** used for repetition detection based on signal peaks.
- **Evaluation**:
  - Mean Absolute Error (MAE): **1.02**

## 🚀 Future Enhancements
- **Real-time form detection** for incorrect posture during exercises.
- **Mobile/Web Application** integration for a seamless user experience.
- **Personalized training recommendations** based on user performance trends.

## 🏆 Results
- **Exercise Classification Accuracy**: **98% (Random Forest)**
- **Rep Counting Accuracy (MAE)**: **1.02**

## 📌 Conclusion
This project demonstrates the **power of AI in fitness tracking**, providing insights into strength training performance. With further improvements, it can serve as an **intelligent fitness assistant** for gym-goers and athletes.

