import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Torch App", page_icon="🔥")

st.title("🔥 PyTorch Demo App")
st.write("This app demonstrates basic PyTorch operations and a small neural network using Streamlit.")

# Section 1: Tensor Operations
st.header("🔢 Tensor Operations")

# Input tensors
a = st.text_input("Enter first tensor values (comma separated):", "1,2,3")
b = st.text_input("Enter second tensor values (comma separated):", "4,5,6")

# Convert input to tensors
try:
    a_tensor = torch.tensor([float(x) for x in a.split(",")])
    b_tensor = torch.tensor([float(x) for x in b.split(",")])

    st.write("**Tensor A:**", a_tensor)
    st.write("**Tensor B:**", b_tensor)

    st.write("**Addition:**", a_tensor + b_tensor)
    st.write("**Multiplication:**", a_tensor * b_tensor)
    st.write("**Dot Product:**", torch.dot(a_tensor, b_tensor))
except Exception as e:
    st.error(f"Error: {e}")

# Section 2: Simple Neural Network
st.header("🧠 Simple Neural Network Example")

# Neural Network Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()

# Input
input_data = st.text_input("Enter 3 input values for neural network:", "0.5, -1.2, 3.3")

try:
    input_tensor = torch.tensor([float(x) for x in input_data.split(",")])
    output = model(input_tensor)
    st.write("**Output:**", output)
except Exception as e:
    st.error(f"Error: {e}")

# Section 3: GPU Info
st.header("💻 GPU Information")
if torch.cuda.is_available():
    st.success("CUDA is available ✅")
    st.write("GPU Name:", torch.cuda.get_device_name(0))
else:
    st.warning("CUDA not available ⚠️ Running on CPU")
