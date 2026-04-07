import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, HBox, VBox, Label, interactive_output
import time

# --- 1. DATA SETUP ---
target_4 = np.zeros((7, 7))
target_4[1:6, 4] = 1.0  # Vertical
target_4[3, 1:5] = 1.0  # Horizontal
target_4[1:4, 1] = 1.0  # Hook
target_score = np.sum(target_4 * target_4)

# --- 2. WIDGETS ---
w_v_s = FloatSlider(min=0, max=1.2, step=0.01, value=0.1, description='Vertical')
w_h_s = FloatSlider(min=0, max=1.2, step=0.01, value=0.1, description='Horizontal')
w_k_s = FloatSlider(min=0, max=1.2, step=0.01, value=0.1, description='Hook')
btn_nudge = Button(description="Backward Pass (Nudge)", button_style='warning')
btn_train = Button(description="Run Until Convergence", button_style='danger')
msg_out = Label(value="System Ready")
grad_out = Label(value="Back-prop signal: 0.00")

# --- 3. THE ARCHITECTURE PLOTTER ---
def draw_network(ax, w_v, w_h, w_k, grad=0):
    inputs = np.linspace(0.2, 0.8, 5)
    hidden = [0.3, 0.5, 0.7]
    h_vals = [w_v, w_h, w_k]

    for i_pos in inputs:
        for j_pos, val in zip(hidden, h_vals):
            ax.plot([0.1, 0.5], [i_pos, j_pos], c='purple', alpha=min(val, 1), lw=max(0.1, val*4))

    for j_pos, val in zip(hidden, h_vals):
        ax.plot([0.5, 0.9], [j_pos, 0.5], c='blue', alpha=0.6, lw=2)
        # Forward Weight
        ax.text(0.65, j_pos + 0.02, f"w={val:.2f}", color='blue', fontsize=9, fontweight='bold')
        # FIXED: Changed 'italic=True' to 'fontstyle='italic''
        if grad != 0:
            ax.text(0.65, j_pos - 0.06, f"Δ={grad:.3f}", color='red', fontsize=9, fontstyle='italic')

    ax.scatter([0.1]*5, inputs, s=150, c='gray', zorder=3)
    ax.scatter([0.5]*3, hidden, s=300, c='purple', zorder=3)
    ax.scatter([0.9], [0.5], s=500, c='blue', zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title("2. Architecture & Signals\n(Blue=Weight, Red=Gradient)")

# --- 4. DASHBOARD RENDER ---
def render_view(w_v, w_h, w_k):
    weights = np.zeros((7, 7))
    weights[1:6, 4] = w_v; weights[3, 1:5] = w_h; weights[1:4, 1] = w_k
    prediction = np.sum(weights * target_4)
    loss = (prediction - target_score)**2
    current_grad = (target_score - prediction) * 0.05 * 0.1

    fig = plt.figure(figsize=(24, 5))
    ax1 = fig.add_subplot(151); ax2 = fig.add_subplot(152)
    ax3 = fig.add_subplot(153); ax4 = fig.add_subplot(154); ax5 = fig.add_subplot(155)

    ax1.imshow(target_4, cmap='Greys'); ax1.set_title("1. Input Data"); ax1.axis('off')
    draw_network(ax2, w_v, w_h, w_k, grad=current_grad)
    ax3.imshow(weights, cmap='Purples', vmin=0, vmax=1.2); ax3.set_title("3. Feature Map")
    ax4.bar(['Target', 'Prediction'], [target_score, prediction], color=['green', 'blue'])
    ax4.set_ylim(0, 15); ax4.set_title("4. Output Score")

    x_range = np.linspace(0, 1.5, 50)
    y_range = (x_range * target_score - target_score)**2
    ax5.plot(x_range, y_range, 'k--', alpha=0.2)
    ax5.scatter((w_v+w_h+w_k)/3, loss, color='red', s=100); ax5.set_title("5. Loss Curve")

    plt.tight_layout(); plt.show()

# --- 5. LOGIC ---
def step_gradient():
    lr = 0.08
    pred = np.sum(target_4) * ((w_v_s.value + w_h_s.value + w_k_s.value) / 3)
    error = target_score - pred
    nudge = error * lr * 0.1
    grad_out.value = f"Back-prop signal (Δ): {nudge:.4f}"
    w_v_s.value += nudge; w_h_s.value += nudge; w_k_s.value += nudge
    return abs(error)

btn_nudge.on_click(lambda b: [step_gradient(), setattr(msg_out, 'value', "Weights updated via back-prop.")])
btn_train.on_click(lambda b: [step_gradient() or time.sleep(0.1) for _ in range(30)])

ui = VBox([HBox([w_v_s, w_h_s, w_k_s]), HBox([btn_nudge, btn_train, msg_out, grad_out])])
out = interactive_output(render_view, {'w_v': w_v_s, 'w_h': w_h_s, 'w_k': w_k_s})
display(ui, out)