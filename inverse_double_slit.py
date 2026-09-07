# inverse_double_slit.py
#
# Fixes applied vs. the original:
#   1. Canonical slit ordering (xL < xR) — removes the swap ambiguity that made
#      the original label vector two-valued for a single input.
#   2. Phase encoded as (cos, sin) instead of a raw angle — removes the
#      0/2π discontinuity that a linear+MSE head cannot represent.
#   3. Targets standardized (known analytic bounds) before training and
#      inverse-transformed before reporting metrics, in physical units.
#   4. generate_dataset() now yields exactly num_samples samples (no silent
#      padding to a multiple of batch_size).
#   5. Models saved in native .keras format instead of legacy .h5.
#   6. tf.data pipeline uses an explicit .repeat() instead of relying on
#      implicit generator re-invocation across epochs.
#   7. Phase error reported as a proper circular (wrapped) error, not raw MSE.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ------------------------------
# 1. Data generation (canonical ordering + cyclic phase target)
# ------------------------------
def gaussian(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

# Analytic bounds used for target normalization (fixed by the sampling ranges below).
X_LO, X_HI = -5.0, 5.0
SIGMA_LO, SIGMA_HI = 0.3, 1.5
A_LO, A_HI = 0.8, 1.2

def _normalize_target(xL, sL, AL, xR, sR, AR, cphi, sphi):
    return np.array([
        (xL - X_LO) / (X_HI - X_LO) * 2 - 1,
        (sL - SIGMA_LO) / (SIGMA_HI - SIGMA_LO) * 2 - 1,
        (AL - A_LO) / (A_HI - A_LO) * 2 - 1,
        (xR - X_LO) / (X_HI - X_LO) * 2 - 1,
        (sR - SIGMA_LO) / (SIGMA_HI - SIGMA_LO) * 2 - 1,
        (AR - A_LO) / (A_HI - A_LO) * 2 - 1,
        cphi,
        sphi,
    ], dtype=np.float32)

def denormalize_targets(y):
    """Inverse of _normalize_target, vectorized over a batch. Returns physical units."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    out[:, 0] = (y[:, 0] + 1) / 2 * (X_HI - X_LO) + X_LO       # xL
    out[:, 1] = (y[:, 1] + 1) / 2 * (SIGMA_HI - SIGMA_LO) + SIGMA_LO  # sigmaL
    out[:, 2] = (y[:, 2] + 1) / 2 * (A_HI - A_LO) + A_LO       # AL
    out[:, 3] = (y[:, 3] + 1) / 2 * (X_HI - X_LO) + X_LO       # xR
    out[:, 4] = (y[:, 4] + 1) / 2 * (SIGMA_HI - SIGMA_LO) + SIGMA_LO  # sigmaR
    out[:, 5] = (y[:, 5] + 1) / 2 * (A_HI - A_LO) + A_LO       # AR
    out[:, 6] = y[:, 6]  # cos(phi)
    out[:, 7] = y[:, 7]  # sin(phi)
    return out

def generate_sample(x_vals, rng):
    # Left slit position; distance kept inside [-5, 5] for both slits.
    xL = rng.uniform(-4, 4)
    max_d = min(5 - xL, xL + 5)
    d = rng.uniform(1.0, min(4.0, max_d))
    xR = xL + d  # always > xL by construction: canonical order, no swap ambiguity

    sigmaL = rng.uniform(SIGMA_LO, SIGMA_HI)
    sigmaR = rng.uniform(SIGMA_LO, SIGMA_HI)
    AL = rng.uniform(A_LO, A_HI)
    AR = rng.uniform(A_LO, A_HI)

    # Only the relative phase is physical; fix the left slit's phase as reference.
    dphi = rng.uniform(0, 2 * np.pi)

    psiL = AL * gaussian(x_vals, xL, sigmaL) * np.exp(1j * 0.0)
    psiR = AR * gaussian(x_vals, xR, sigmaR) * np.exp(1j * dphi)
    intensity = np.abs(psiL + psiR) ** 2
    intensity = intensity / np.max(intensity + 1e-8)

    target = _normalize_target(xL, sigmaL, AL, xR, sigmaR, AR,
                                np.cos(dphi), np.sin(dphi))
    return intensity.astype(np.float32), target

def generate_dataset(num_samples, x_vals, batch_size, seed=42):
    """Yields exactly num_samples samples total, batched (last batch may be smaller)."""
    rng = np.random.default_rng(seed)
    def gen():
        remaining = num_samples
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            X_batch, y_batch = [], []
            for _ in range(this_batch):
                I, s = generate_sample(x_vals, rng)
                X_batch.append(I)
                y_batch.append(s)
            remaining -= this_batch
            yield np.array(X_batch), np.array(y_batch)
    return gen

# ------------------------------
# 2. Model
# ------------------------------
def build_inverse_model(input_dim=100, output_dim=8):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='linear'),
    ])
    return model

# ------------------------------
# 3. Data setup
# ------------------------------
x_vals = np.linspace(-10, 10, 100)
batch_size = 64
train_samples = 200_000
val_samples = 10_000
test_samples = 10_000

val_gen = generate_dataset(val_samples, x_vals, batch_size, seed=123)
X_val, y_val = map(np.concatenate, zip(*[(Xb, yb) for Xb, yb in val_gen()]))

test_gen = generate_dataset(test_samples, x_vals, batch_size, seed=456)
X_test, y_test = map(np.concatenate, zip(*[(Xb, yb) for Xb, yb in test_gen()]))

train_gen = generate_dataset(train_samples, x_vals, batch_size, seed=42)

# ------------------------------
# 4. Training
# ------------------------------
model = build_inverse_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mse',
              metrics=['mae'])

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
]

train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
    ),
).repeat().prefetch(tf.data.AUTOTUNE)  # explicit repeat: don't rely on implicit re-invocation

history = model.fit(
    train_ds,
    epochs=100,
    steps_per_epoch=500,
    validation_data=(X_val, y_val),
    callbacks=callbacks_list,
    verbose=1,
)

# ------------------------------
# 5. Evaluation (in physical units, with proper circular phase error)
# ------------------------------
y_pred_norm = model.predict(X_test)
y_test_phys = denormalize_targets(y_test)
y_pred_phys = denormalize_targets(y_pred_norm)

# Recover phase angles from (cos, sin) via atan2, then wrap the error to [-pi, pi].
phi_true = np.arctan2(y_test_phys[:, 7], y_test_phys[:, 6])
phi_pred = np.arctan2(y_pred_phys[:, 7], y_pred_phys[:, 6])
phi_err = np.angle(np.exp(1j * (phi_pred - phi_true)))  # wrapped difference

param_names = ['xL', 'sigmaL', 'AL', 'xR', 'sigmaR', 'AR']
print("Per-parameter MAE (physical units):")
for i, name in enumerate(param_names):
    mae = np.mean(np.abs(y_test_phys[:, i] - y_pred_phys[:, i]))
    print(f"  {name:8s}: {mae:.4f}")
print(f"  phase   : {np.mean(np.abs(phi_err)):.4f} rad (circular MAE)")

r2 = r2_score(y_test, y_pred_norm, multioutput='variance_weighted')
print(f"Normalized-target R^2: {r2:.4f}")

# ------------------------------
# 6. Plots
# ------------------------------
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE (normalized targets)')
plt.legend()
plt.title('Training History')
plt.savefig('loss_curve.png')
plt.show()

idx = np.random.choice(len(X_test), 5, replace=False)
plt.figure(figsize=(12, 8))
for i, j in enumerate(idx):
    plt.subplot(2, 3, i + 1)
    plt.plot(x_vals, X_test[j], label='True intensity')

    xLp, sLp, ALp, xRp, sRp, ARp, cphip, sphip = y_pred_phys[j]
    phip = np.arctan2(sphip, cphip)
    psiLp = ALp * gaussian(x_vals, xLp, sLp) * np.exp(1j * 0.0)
    psiRp = ARp * gaussian(x_vals, xRp, sRp) * np.exp(1j * phip)
    I_pred = np.abs(psiLp + psiRp) ** 2
    I_pred = I_pred / (np.max(I_pred) + 1e-8)

    plt.plot(x_vals, I_pred, '--', label='Predicted intensity')
    plt.title(f"Sample {j}")
    plt.legend()
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()

model.save('inverse_double_slit_final.keras')
print("Done. Model saved.")
