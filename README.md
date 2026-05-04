# 🧪 Lab 4: RNN vs LSTM vs GRU Comparison

## 📌 Objective

To compare the performance of **RNN, LSTM, and GRU** models on sequential data and analyze their efficiency in terms of accuracy, training time, and memory handling.

---

## 🧠 Introduction

Recurrent Neural Networks (RNN) and their advanced variants like LSTM and GRU are widely used for handling sequential data such as:

* Text
* Time-series
* Speech

However, basic RNN suffers from the **vanishing gradient problem**, which limits its ability to learn long-term dependencies. LSTM and GRU are designed to overcome this issue.

---

## ⚙️ Models Used

### 🔹 RNN (Recurrent Neural Network)

* Simple architecture
* Short-term memory
* Fast but less accurate

### 🔹 LSTM (Long Short-Term Memory)

* Uses memory cells and gates
* Handles long-term dependencies
* High accuracy but slower

### 🔹 GRU (Gated Recurrent Unit)

* Simplified LSTM
* Fewer gates
* Faster than LSTM with similar performance

---

## 📊 Comparison Table

| Feature            | RNN        | LSTM      | GRU                  |
| ------------------ | ---------- | --------- | -------------------- |
| Memory             | Short-term | Long-term | Long-term            |
| Vanishing Gradient | Yes ❌      | No ✅      | No ✅                 |
| Complexity         | Low        | High      | Medium               |
| Training Time      | Fast       | Slow      | Faster than LSTM     |
| Accuracy           | Low        | High      | Almost equal to LSTM |
| Gates              | None       | 3 Gates   | 2 Gates              |

---

## 🧪 Experiment Details

### 📁 Dataset

* Time-series data / Text dataset (e.g., IMDB or synthetic data)

### 🔄 Steps Performed

1. Load dataset
2. Preprocess data (normalization / tokenization)
3. Build three models:

   * Simple RNN
   * LSTM
   * GRU
4. Train all models using same parameters
5. Compare:

   * Accuracy
   * Loss
   * Training Time

---

## 💻 Sample Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

def build_model(model_type):
    model = Sequential()
    
    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(10, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(10, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(10, 1)))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model
```

---

## 📈 Expected Results

* **RNN** → Fast training but poor performance on long sequences
* **LSTM** → Best accuracy but slower training
* **GRU** → Balanced performance (good accuracy + faster than LSTM)

---

## 🎯 Conclusion

* RNN is not suitable for long sequence tasks due to vanishing gradient problem
* LSTM provides the best performance by capturing long-term dependencies
* GRU offers a good trade-off between speed and accuracy
* For real-world applications, **LSTM and GRU are preferred**

---

## 🚀 Future Improvements

* Add dropout layers to avoid overfitting
* Use larger datasets
* Tune hyperparameters for better accuracy
* Visualize training graphs (accuracy vs loss)

---

## 📌 Author

**Om Prakash Kannaujiya**

---
