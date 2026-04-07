import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. SESSION STATE (The App's Memory) ---
# This ensures the weights don't reset every time you click a button
if 'w_v' not in st.session_state:
    st.session_state.w_v = 0.1
    st.session_state.w_h = 0.1
    st.session_state.w_k = 0.1

# --- 2. DATA SETUP ---
target_4 = np.zeros((7, 7))
target_4[1:6, 4] = 1.0  # Vertical
target_4[3, 1:5] = 1.0  # Horizontal
target_4[1:4, 1] = 1.0  # Hook
target_score = np.sum(target_4 * target_4)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Deep Learning Visualizer", layout="wide")
st.title("Interactive Neural Network: Learning the Digit '4'")
st.write("Adjust the features manually or trigger Backpropagation to see the weights update.")

# Sidebar for Sliders
st.sidebar.header("Manual Weight Control")
st.session_state.w_v = st.sidebar.slider("Vertical Stroke", 0.0, 1.2, st.session_state.w_v)
st.session_state.w_h = st.sidebar.slider("Horizontal Stroke", 0.0, 1.2, st.session_state.w_h)
st.session_state.w_k = st.sidebar.slider("Hook Stroke", 0.0, 1.2, st.session_state.w_k)

# --- 4. LEARNING LOGIC ---
def step_gradient():
    lr = 0.08
    current_avg = (st.session_state.w_v + st.session_state.w_h + st.session_state.w_k) / 3
    error = target_score - (np.sum(target_4) * current_avg)
    nudge = error * lr * 0.1
    st.session_state.w_v += nudge
    st.session_state.w_h += nudge
    st.session_state.w_k += nudge
    return nudge

# Buttons for Training
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Backward Pass (Nudge)"):
    nudge_val = step_gradient()
    st.sidebar.write(f"Back-prop Nudge: {nudge_val:.4f}")

if col_btn2.button("Run to Convergence"):
    for _ in range(30):
        step_gradient()
    st.balloons()

# --- 5. VISUALIZATION FUNCTION ---
def draw_network(ax, w_v, w_h, w_k):
    inputs = np.linspace(0.2, 0.8, 5)
    hidden = [0.3, 0.5, 0.7]
    h_vals = [w_v, w_h, w_k]
    
    for i_pos in inputs:
        for j_pos, val in zip(hidden, h_vals):
            ax.plot([0.1, 0.5], [i_pos, j_pos], c='purple', alpha=min(val, 1), lw=max(0.1, val*4))
            
    for j_pos, val in zip(hidden, h_vals):
        ax.plot([0.5, 0.9], [j_pos, 0.5], c='blue', alpha=0.6, lw=2)
        ax.text(0.65, j_pos + 0.02, f"w={val:.2f}", color='blue', fontsize=9, fontweight='bold')

    ax.scatter([0.1]*5, inputs, s=150, c='gray', zorder=3)
    ax.scatter([0.5]*3, hidden, s=300, c='purple', zorder=3)
    ax.scatter([0.9], [0.5], s=500, c='blue', zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# --- 6. RENDER DASHBOARD ---
w_v, w_h, w_k = st.session_state.w_v, st.session_state.w_h, st.session_state.w_k
weights = np.zeros((7, 7))
weights[1:6, 4] = w_v; weights[3, 1:5] = w_h; weights[1:4, 1] = w_k
prediction = np.sum(weights * target_4)
loss = (prediction - target_score)**2

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

axes[0].imshow(target_4, cmap='Greys'); axes[0].set_title("1. Input Data"); axes[0].axis('off')
draw_network(axes[1], w_v, w_h, w_k); axes[1].set_title("2. Architecture")
axes[2].imshow(weights, cmap='Purples', vmin=0, vmax=1.2); axes[2].set_title("3. Internal Features")
axes[3].bar(['Target', 'Prediction'], [target_score, prediction], color=['green', 'blue'])
axes[3].set_ylim(0, 15); axes[3].set_title("4. Output Score")
x_range = np.linspace(0, 1.5, 50)
y_range = (x_range * target_score - target_score)**2
axes[4].plot(x_range, y_range, 'k--', alpha=0.2)
axes[4].scatter((w_v+w_h+w_k)/3, loss, color='red', s=100); axes[4].set_title("5. Loss Curve")

st.pyplot(fig)