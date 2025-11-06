import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# === Page Configuration ===
st.set_page_config(page_title="MCA572 Lab 2 - Activation Functions", page_icon="🧠")

# === Activation Functions ===
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid_binary(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_bipolar(x):
    return (2 / (1 + np.exp(-np.clip(x, -250, 250)))) - 1

def tanh_function(x):
    return np.tanh(x)

def relu_function(x):
    return np.maximum(0, x)

# === Neural Network Class ===
class SimpleNeuralNetwork:
    def __init__(self, activation_func, learning_rate=0.1):
        self.activation = activation_func
        self.lr = learning_rate
        # Initialize weights randomly
        np.random.seed(42)
        self.W1 = np.random.randn(2, 4) * 0.1  # Input to hidden
        self.b1 = np.zeros((1, 4))             # Hidden bias
        self.W2 = np.random.randn(4, 1) * 0.1  # Hidden to output
        self.b2 = np.zeros((1, 1))             # Output bias
    
    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        # Output layer (always sigmoid)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid_binary(self.z2)
        return self.a2
    
    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass (simplified)
            error = y - output
            
            # Update weights (simplified gradient descent)
            d_output = error * output * (1 - output)
            
            # Update output layer
            self.W2 += self.a1.T @ d_output * self.lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            
            # Update hidden layer  
            d_hidden = (d_output @ self.W2.T) * (self.a1 > 0)  # Simplified derivative
            self.W1 += X.T @ d_hidden * self.lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# === Streamlit App ===
st.title("🧠 MCA572 - Lab 2: Activation Functions")
st.markdown("**Simple demonstration of activation functions and neural network training**")

# === Section 1: Activation Function Visualization ===
st.header("1. Activation Functions Visualization")

# Generate input range
x = np.linspace(-5, 5, 100)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Activation Functions', fontsize=14)

# Step Function
axes[0,0].plot(x, step_function(x), 'b-', linewidth=2)
axes[0,0].set_title('Step Function')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim(-0.5, 1.5)

# Binary Sigmoid
axes[0,1].plot(x, sigmoid_binary(x), 'r-', linewidth=2)
axes[0,1].set_title('Binary Sigmoid [0,1]')
axes[0,1].grid(True, alpha=0.3)

# Bipolar Sigmoid
axes[0,2].plot(x, sigmoid_bipolar(x), 'g-', linewidth=2)
axes[0,2].set_title('Bipolar Sigmoid [-1,1]')
axes[0,2].grid(True, alpha=0.3)

# Tanh
axes[1,0].plot(x, tanh_function(x), 'm-', linewidth=2)
axes[1,0].set_title('Tanh Function')
axes[1,0].grid(True, alpha=0.3)

# ReLU
axes[1,1].plot(x, relu_function(x), 'c-', linewidth=2)
axes[1,1].set_title('ReLU Function')
axes[1,1].grid(True, alpha=0.3)

# All functions comparison
axes[1,2].plot(x, sigmoid_binary(x), 'r-', label='Sigmoid', linewidth=2)
axes[1,2].plot(x, tanh_function(x), 'm-', label='Tanh', linewidth=2)
axes[1,2].plot(x, relu_function(x), 'c-', label='ReLU', linewidth=2)
axes[1,2].set_title('Comparison')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# === Section 2: Neural Network Training ===
st.header("2. Neural Network Training on XOR Problem")

# XOR Dataset
st.subheader("XOR Truth Table")
col1, col2 = st.columns(2)

with col1:
    st.write("**Input → Output**")
    st.write("0, 0 → 0")
    st.write("0, 1 → 1") 
    st.write("1, 0 → 1")
    st.write("1, 1 → 0")

with col2:
    activation_choice = st.selectbox(
        "Choose Activation Function:",
        ["Sigmoid", "Tanh", "ReLU"]
    )
    
    epochs = st.slider("Training Epochs:", 100, 2000, 1000, 100)

# Map activation functions
activation_map = {
    "Sigmoid": sigmoid_binary,
    "Tanh": tanh_function, 
    "ReLU": relu_function
}

# Train button
if st.button("🚀 Train Neural Network"):
    # XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    activation_func = activation_map[activation_choice]
    nn = SimpleNeuralNetwork(activation_func)
    
    with st.spinner(f"Training with {activation_choice} activation..."):
        nn.train(X, y, epochs=epochs)
    
    # Get predictions
    predictions = nn.predict(X)
    raw_outputs = nn.forward(X)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y) * 100
    
    # Display results
    st.success("Training completed!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Results")
        for i in range(4):
            input_a, input_b = X[i]
            expected = y[i][0]
            predicted = predictions[i][0]
            raw_output = raw_outputs[i][0]
            
            status = "✅" if predicted == expected else "❌"
            st.write(f"({input_a}, {input_b}) → Expected: {expected}, Got: {predicted} ({raw_output:.3f}) {status}")
    
    with col2:
        st.subheader("Performance")
        st.metric("Accuracy", f"{accuracy:.1f}%")
        st.metric("Activation", activation_choice)
        st.metric("Epochs", epochs)
        
        if accuracy == 100:
            st.success("🎉 Perfect! Network learned XOR!")
        elif accuracy >= 75:
            st.warning("⚠️ Good but not perfect")
        else:
            st.error("❌ Poor performance")

# === Section 3: Performance Comparison ===
st.header("3. Performance Comparison")

if st.button("📊 Compare All Activation Functions"):
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    results = []
    
    for name, func in activation_map.items():
        accuracies = []
        
        # Test 3 times for each activation
        for _ in range(3):
            nn = SimpleNeuralNetwork(func)
            nn.train(X, y, epochs=1000)
            preds = nn.predict(X)
            acc = np.mean(preds == y) * 100
            accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies)
        results.append((name, avg_accuracy))
    
    # Display comparison
    st.subheader("Comparison Results")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    
    # Create bars with different colors
    bars = ax.bar(names, accuracies)
    
    # Customize the chart
    ax.set_title('Activation Functions Performance Comparison')
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Display the chart
    st.pyplot(fig)
    
    # Display text results
    for name, accuracy in results:
        st.write(f"**{name}**: {accuracy:.1f}% average accuracy")
    
    # Find best performer
    best_activation = max(results, key=lambda x: x[1])
    st.success(f"🏆 Best performer: {best_activation[0]} with {best_activation[1]:.1f}% accuracy")

