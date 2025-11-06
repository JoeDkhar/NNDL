import streamlit as st
import os
import urllib.request
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def download_and_load_cifar10():
    CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    DOWNLOAD_DIR = "cifar_data"
    ARCHIVE_PATH = os.path.join(DOWNLOAD_DIR, "cifar-10-python.tar.gz")
    EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "cifar-10-batches-py")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        urllib.request.urlretrieve(CIFAR_URL, ARCHIVE_PATH)
    if not os.path.exists(EXTRACT_DIR):
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)

    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        return d

    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = load_batch(os.path.join(EXTRACT_DIR, f"data_batch_{i}"))
        train_data.append(batch['data'])
        train_labels += batch['labels']
    X_train = np.vstack(train_data).astype(np.float32)
    y_train = np.array(train_labels, dtype=np.int32)

    test_batch = load_batch(os.path.join(EXTRACT_DIR, "test_batch"))
    X_test = test_batch['data'].astype(np.float32)
    y_test = np.array(test_batch['labels'], dtype=np.int32)

    return X_train, y_train, X_test, y_test


def to_one_hot(y, num_classes=10):
    one = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one[np.arange(y.shape[0]), y] = 1.0
    return one


def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(y): return y * (1 - y)
def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def categorical_crossentropy(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def init_params(sizes, rng):
    params = {}
    for i in range(len(sizes) - 1):
        limit = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
        params[f"W{i+1}"] = rng.uniform(-limit, limit, (sizes[i], sizes[i + 1]))
        params[f"b{i+1}"] = np.zeros((1, sizes[i + 1]))
    return params


def forward(params, X):
    cache = {'A0': X}
    L = len([k for k in params if k.startswith("W")])
    for i in range(1, L + 1):
        Z = cache[f"A{i-1}"].dot(params[f"W{i}"]) + params[f"b{i}"]
        A = sigmoid(Z) if i < L else softmax(Z)
        cache[f"Z{i}"], cache[f"A{i}"] = Z, A
    return cache


def backward(params, cache, Y):
    grads = {}
    L = len([k for k in params if k.startswith("W")])
    m = Y.shape[0]
    dZ = (cache[f"A{L}"] - Y) / m

    for i in reversed(range(1, L + 1)):
        grads[f"dW{i}"] = cache[f"A{i-1}"].T.dot(dZ)
        grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)
        if i > 1:
            dA_prev = dZ.dot(params[f"W{i}"].T)
            dZ = dA_prev * sigmoid_deriv(cache[f"A{i-1}"])
    return grads


def update_params(params, grads, lr):
    L = len([k for k in params if k.startswith("W")])
    for i in range(1, L + 1):
        params[f"W{i}"] -= lr * grads[f"dW{i}"]
        params[f"b{i}"] -= lr * grads[f"db{i}"]
    return params


def train_model(X, Y, X_val, Y_val, lr=0.05, epochs=20, batch_size=128):
    input_dim = X.shape[1]
    hidden_sizes = [512, 256, 128]
    output_size = Y.shape[1]
    sizes = [input_dim] + hidden_sizes + [output_size]

    rng = np.random.RandomState(42)
    params = init_params(sizes, rng)
    progress = st.progress(0)
    loss_placeholder = st.empty()

    for epoch in range(epochs):
        idx = rng.permutation(X.shape[0])
        X, Y = X[idx], Y[idx]
        losses = []

        for i in range(0, X.shape[0], batch_size):
            Xb, Yb = X[i:i + batch_size], Y[i:i + batch_size]
            cache = forward(params, Xb)
            loss = categorical_crossentropy(cache[f"A{len(sizes)-1}"], Yb)
            grads = backward(params, cache, Yb)
            params = update_params(params, grads, lr)
            losses.append(loss)

        preds_val = np.argmax(forward(params, X_val)[f"A{len(sizes)-1}"], axis=1)
        acc_val = accuracy_score(np.argmax(Y_val, axis=1), preds_val)
        loss_placeholder.write(f"Epoch {epoch+1}/{epochs} — Loss: {np.mean(losses):.4f} — Val Acc: {acc_val:.4f}")
        progress.progress((epoch + 1) / epochs)
    return params


def predict(params, X):
    return np.argmax(forward(params, X)[f"A{len([k for k in params if k.startswith('W')])}"], axis=1)


# ---------------------------
# Streamlit UI (fixed with session state)
# ---------------------------
st.title(" CIFAR-10 Neural Network Training App")

# Initialize session state
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

if st.button(" Load CIFAR-10 Dataset"):
    X_train, y_train, X_test, y_test = download_and_load_cifar10()
    X_train /= 255.0
    X_test /= 255.0
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.dataset_loaded = True
    st.success(" CIFAR-10 dataset loaded successfully!")

# Only show train button if dataset loaded
if st.session_state.dataset_loaded:
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    st.write("Training Data Shape:", X_train.shape)
    st.write("Test Data Shape:", X_test.shape)

    if st.button(" Train Model (20 Epochs, SGD + Cross-Entropy)"):
        with st.spinner("Training neural network..."):
            params = train_model(X_train, Y_train, X_test, Y_test, lr=0.05, epochs=20, batch_size=128)

        preds = predict(params, X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        cm = confusion_matrix(y_test, preds)

        st.subheader(" Evaluation Metrics")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-Score:** {f1:.4f}")

        st.subheader("🌀 Confusion Matrix")
        st.dataframe(cm)

        st.subheader(" Sample Classified Images")
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        idx = np.random.choice(len(X_test), 10, replace=False)
        for i, ax in enumerate(axes.flat):
            img = X_test[idx[i]].reshape(3, 32, 32).transpose(1, 2, 0)
            true, pred = class_names[y_test[idx[i]]], class_names[preds[idx[i]]]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"T:{true}\nP:{pred}", color="green" if true == pred else "red", fontsize=9)
        st.pyplot(fig)
        st.balloons()
