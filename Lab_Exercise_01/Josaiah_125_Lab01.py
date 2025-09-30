import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# === Streamlit Configuration ===
st.set_page_config(
    page_title="Single Layer Perceptron - Logic Gates Lab",
    page_icon="🧠",
    layout="centered"
)

# === Truth Tables ===
truth_tables = {
    "AND": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,0,1])),
    "OR": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,1])),
    "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0])),
    "AND-NOT": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,1,0]))
}

# === Activation Functions ===
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# === Simple Perceptron Class ===
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, weight_init="random"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        if weight_init == "defined":
            self.weights = np.array([0.5, 0.5])
            self.bias = 0.0
        else:
            self.weights = np.random.randn(input_size) * 0.1
            self.bias = np.random.randn() * 0.1
            
        # Store initial values for comparison
        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return sigmoid(linear_output)

    def train(self, X, y):
        training_history = []
        weight_history = []
        
        for epoch in range(self.epochs):
            total_error = 0
            
            for xi, target in zip(X, y):
                # Forward pass
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = sigmoid(linear_output)
                
                # Calculate error
                error = target - prediction
                
                # Backward pass (gradient descent)
                gradient = error * sigmoid_derivative(linear_output)
                
                # Update weights and bias
                self.weights += self.learning_rate * gradient * xi
                self.bias += self.learning_rate * gradient
                
                total_error += abs(error)
            
            training_history.append(total_error)
            weight_history.append([self.weights.copy(), self.bias])
            
        return training_history, weight_history

# === Main Application ===
st.title("🧠 Single Layer Perceptron - Logic Gates Lab")
st.markdown("**Lab Exercise**: Building Neural Networks to Simulate Logic Gates")

# === Gate Selection ===
st.subheader("1. Select Logic Gate")
gate = st.selectbox("Choose a Logic Gate:", list(truth_tables.keys()))

# Display truth table
st.subheader("2. Truth Table")
X, y = truth_tables[gate]
st.write("Input-Output Combinations:")

truth_table_data = []
for i in range(len(X)):
    truth_table_data.append([f"Input A: {X[i][0]}, Input B: {X[i][1]}", f"Output: {y[i]}"])

for row in truth_table_data:
    st.write(f"• {row[0]} → {row[1]}")

# === Model Configuration ===
st.subheader("3. Perceptron Configuration")
col1, col2 = st.columns(2)

with col1:
    weight_init = st.radio("Weight Initialization:", ["random", "defined"])
    learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, 0.01)

with col2:
    epochs = st.slider("Training Epochs:", 10, 500, 100, 10)

# === Training Section ===
st.subheader("4. Train the Perceptron")

if st.button("🚀 Train Model", type="primary"):
    # Create and train the model
    model = Perceptron(input_size=2, learning_rate=learning_rate, epochs=epochs, weight_init=weight_init)
    
    with st.spinner("Training in progress..."):
        training_history, weight_history = model.train(X, y)
    
    st.success("Training completed!")
    
    # === Results Section ===
    st.subheader("5. Results & Analysis")
    
    # Training Progress
    st.write("**Training Error Over Time:**")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Error')
    ax.set_title(f'{gate} Gate - Training Progress')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Predictions
    st.write("**Final Predictions:**")
    for i, (xi, target) in enumerate(zip(X, y)):
        prediction = model.predict(xi)
        binary_pred = 1 if prediction > 0.5 else 0
        accuracy = "✅ Correct" if binary_pred == target else "❌ Wrong"
        
        st.write(f"Input: [{xi[0]}, {xi[1]}] → Predicted: {prediction:.3f} ({binary_pred}) | Expected: {target} | {accuracy}")
    
    # Weight Analysis
    st.write("**Weight Changes During Training:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Initial Values:**")
        st.write(f"• Weight A: {model.initial_weights[0]:.3f}")
        st.write(f"• Weight B: {model.initial_weights[1]:.3f}")
        st.write(f"• Bias: {model.initial_bias:.3f}")
    
    with col2:
        st.write("**Final Values:**")
        st.write(f"• Weight A: {model.weights[0]:.3f}")
        st.write(f"• Weight B: {model.weights[1]:.3f}")
        st.write(f"• Bias: {model.bias:.3f}")
    
    # Calculate accuracy
    correct = 0
    for xi, target in zip(X, y):
        prediction = model.predict(xi)
        if (prediction > 0.5 and target == 1) or (prediction <= 0.5 and target == 0):
            correct += 1
    
    accuracy = (correct / len(X)) * 100
    
    if accuracy == 100:
        st.success(f"🎉 Perfect! The perceptron learned the {gate} gate with {accuracy:.1f}% accuracy!")
    elif accuracy >= 75:
        st.warning(f"⚠️ Good performance: {accuracy:.1f}% accuracy for {gate} gate")
    else:
        st.error(f"❌ Poor performance: {accuracy:.1f}% accuracy for {gate} gate")
    
    # Special note for XOR
    if gate == "XOR" and accuracy < 100:
        st.error("🔍 **Important Observation**: The XOR gate is not linearly separable!")
        st.write("**Why does the Single Layer Perceptron struggle with XOR?**")
        st.write("• XOR requires a non-linear decision boundary")
        st.write("• A single layer perceptron can only create linear boundaries")
        st.write("• Solution: Use a Multi-Layer Perceptron (MLP) with hidden layers")

# === Lab Questions Section ===
st.markdown("---")
st.subheader("📝 Lab Questions to Consider")

gate_questions = {
    "AND": [
        "How do the weights and bias values change during training for the AND gate?",
        "Can the perceptron successfully learn the AND logic with a linear decision boundary?"
    ],
    "OR": [
        "What changes in the perceptron's weights are necessary to represent the OR gate logic?",
        "How does the linear decision boundary look for the OR gate classification?"
    ],
    "AND-NOT": [
        "What is the perceptron's weight configuration after training for the AND-NOT gate?",
        "How does the perceptron handle cases where both inputs are 1 or 0?"
    ],
    "XOR": [
        "Why does the Single Layer Perceptron struggle to classify the XOR gate?",
        "What modifications can be made to the neural network model to handle the XOR gate correctly?"
    ]
}

if gate in gate_questions:
    for i, question in enumerate(gate_questions[gate], 1):
        st.write(f"**Q{i}.** {question}")

# === Footer ===
st.markdown("---")
st.markdown("""
**Lab Exercise Goals:**
- Understand how Single Layer Perceptrons work
- Learn the limitations of linear classifiers
- Observe weight updates during training
- Analyze the XOR problem and linear separability
""")