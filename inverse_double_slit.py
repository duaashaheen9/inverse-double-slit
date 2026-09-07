# inverse_double_slit.py
#
# Inverse reconstruction of a simplified two-slit interference model
# using Deep Learning with TensorFlow/Keras.
#
# Main design decisions:
#   1. Canonical slit ordering: x_left < x_right.
#   2. Only relative phase is modeled; the left slit is the phase reference.
#   3. The intensity is NOT normalized, so the absolute intensity scale
#      is preserved.
#   4. For this simplified real-valued Gaussian model, intensity depends
#      on relative phase through cos(delta_phi). Therefore cos(delta_phi)
#      is used as the identifiable phase-related target.
#   5. Physical parameters are scaled to approximately [-1, 1] for training
#      and transformed back to physical/model units for evaluation.
#   6. The data generator yields exactly the requested number of samples.
#   7. The training tf.data pipeline explicitly repeats the generator.
#
# NOTE:
# This is a simplified computational model for inverse scientific learning.
# It is not intended to be a full physical simulation of an experimental
# double-slit quantum system.


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# ============================================================
# 1. Physical / Model Parameters
# ============================================================

# Spatial domain used to represent the measured intensity pattern.
X_MIN = -10.0
X_MAX = 10.0
NUM_POINTS = 100

# Parameter ranges used during synthetic data generation.
X_LO, X_HI = -5.0, 5.0
SIGMA_LO, SIGMA_HI = 0.3, 1.5
A_LO, A_HI = 0.8, 1.2


# ============================================================
# 2. Forward Physics Model
# ============================================================

def gaussian(x, center, sigma):
    """
    Real-valued Gaussian spatial envelope.

    Parameters
    ----------
    x : np.ndarray
        Spatial measurement positions.
    center : float
        Center position of the Gaussian.
    sigma : float
        Width of the Gaussian.

    Returns
    -------
    np.ndarray
        Gaussian evaluated at x.
    """
    return np.exp(
        -((x - center) ** 2) / (2.0 * sigma ** 2)
    )


def compute_intensity(
    x_vals,
    x_left,
    x_right,
    sigma_left,
    sigma_right,
    A_left,
    A_right,
    delta_phi,
):
    """
    Compute the raw two-slit interference intensity.

    The complex wave amplitudes are:

        psi_left  = A_left  * G_left
        psi_right = A_right * G_right * exp(i * delta_phi)

    and the observable intensity is:

        I(x) = |psi_left + psi_right|^2

    No normalization is applied.
    """

    psi_left = (
        A_left
        * gaussian(x_vals, x_left, sigma_left)
    )

    psi_right = (
        A_right
        * gaussian(x_vals, x_right, sigma_right)
        * np.exp(1j * delta_phi)
    )

    intensity = np.abs(
        psi_left + psi_right
    ) ** 2

    return intensity.astype(np.float32)


def compute_intensity_from_cos_phase(
    x_vals,
    x_left,
    x_right,
    sigma_left,
    sigma_right,
    A_left,
    A_right,
    cos_delta_phi,
):
    """
    Equivalent real-valued form of the intensity model.

    For the present model:

        I(x) =
            A_left^2  G_left^2
          + A_right^2 G_right^2
          + 2 A_left A_right G_left G_right cos(delta_phi)

    This form is useful during parameter reconstruction because
    cos(delta_phi) is the identifiable phase-related quantity.
    """

    G_left = gaussian(
        x_vals,
        x_left,
        sigma_left
    )

    G_right = gaussian(
        x_vals,
        x_right,
        sigma_right
    )

    intensity = (
        (A_left ** 2) * (G_left ** 2)
        + (A_right ** 2) * (G_right ** 2)
        + 2.0
        * A_left
        * A_right
        * G_left
        * G_right
        * cos_delta_phi
    )

    return intensity.astype(np.float32)


# ============================================================
# 3. Target Scaling
# ============================================================

def normalize_target(
    x_left,
    x_right,
    sigma_left,
    sigma_right,
    A_left,
    A_right,
    cos_delta_phi,
):
    """
    Scale physical parameters to approximately [-1, 1].

    Target order:

        0: x_left
        1: x_right
        2: sigma_left
        3: sigma_right
        4: A_left
        5: A_right
        6: cos(delta_phi)

    The phase-related target is already naturally in [-1, 1].
    """

    return np.array([
        (x_left - X_LO)
        / (X_HI - X_LO) * 2.0 - 1.0,

        (x_right - X_LO)
        / (X_HI - X_LO) * 2.0 - 1.0,

        (sigma_left - SIGMA_LO)
        / (SIGMA_HI - SIGMA_LO) * 2.0 - 1.0,

        (sigma_right - SIGMA_LO)
        / (SIGMA_HI - SIGMA_LO) * 2.0 - 1.0,

        (A_left - A_LO)
        / (A_HI - A_LO) * 2.0 - 1.0,

        (A_right - A_LO)
        / (A_HI - A_LO) * 2.0 - 1.0,

        cos_delta_phi,

    ], dtype=np.float32)


def denormalize_targets(y):
    """
    Transform normalized targets back to physical/model units.

    Supports either a single batch of predictions or test targets.

    Output order:

        x_left
        x_right
        sigma_left
        sigma_right
        A_left
        A_right
        cos(delta_phi)
    """

    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 2 or y.shape[1] != 7:
        raise ValueError(
            "Expected y with shape (N, 7). "
            f"Received shape: {y.shape}"
        )

    out = np.empty_like(y)

    out[:, 0] = (
        (y[:, 0] + 1.0) / 2.0
        * (X_HI - X_LO)
        + X_LO
    )

    out[:, 1] = (
        (y[:, 1] + 1.0) / 2.0
        * (X_HI - X_LO)
        + X_LO
    )

    out[:, 2] = (
        (y[:, 2] + 1.0) / 2.0
        * (SIGMA_HI - SIGMA_LO)
        + SIGMA_LO
    )

    out[:, 3] = (
        (y[:, 3] + 1.0) / 2.0
        * (SIGMA_HI - SIGMA_LO)
        + SIGMA_LO
    )

    out[:, 4] = (
        (y[:, 4] + 1.0) / 2.0
        * (A_HI - A_LO)
        + A_LO
    )

    out[:, 5] = (
        (y[:, 5] + 1.0) / 2.0
        * (A_HI - A_LO)
        + A_LO
    )

    out[:, 6] = y[:, 6]

    return out


# ============================================================
# 4. Synthetic Data Generation
# ============================================================

def generate_sample(x_vals, rng):
    """
    Generate one synthetic two-slit interference sample.

    The slit positions are generated directly so that:

        -5 <= x_left < x_right <= 5
        1 <= x_right - x_left <= 4

    Returns
    -------
    intensity : np.ndarray, shape (NUM_POINTS,)
        Raw, non-normalized intensity.

    target : np.ndarray, shape (7,)
        Normalized target parameters.
    """

    # --------------------------------------------------------
    # Slit geometry
    # --------------------------------------------------------

    # Generate the distance first.
    distance = rng.uniform(1.0, 4.0)

    # Then choose a left position that guarantees
    # x_right = x_left + distance <= 5.
    x_left = rng.uniform(
        X_LO,
        X_HI - distance
    )

    x_right = x_left + distance

    # --------------------------------------------------------
    # Gaussian widths
    # --------------------------------------------------------

    sigma_left = rng.uniform(
        SIGMA_LO,
        SIGMA_HI
    )

    sigma_right = rng.uniform(
        SIGMA_LO,
        SIGMA_HI
    )

    # --------------------------------------------------------
    # Amplitudes
    # --------------------------------------------------------

    A_left = rng.uniform(
        A_LO,
        A_HI
    )

    A_right = rng.uniform(
        A_LO,
        A_HI
    )

    # --------------------------------------------------------
    # Relative phase
    # --------------------------------------------------------

    # The absolute phase is not observable from intensity,
    # so the left slit is used as the phase reference:
    #
    #     phi_left = 0
    #
    # and only the relative phase is sampled.
    delta_phi = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    # --------------------------------------------------------
    # Forward model
    # --------------------------------------------------------

    intensity = compute_intensity(
        x_vals=x_vals,
        x_left=x_left,
        x_right=x_right,
        sigma_left=sigma_left,
        sigma_right=sigma_right,
        A_left=A_left,
        A_right=A_right,
        delta_phi=delta_phi,
    )

    # --------------------------------------------------------
    # Identifiable target representation
    # --------------------------------------------------------

    cos_delta_phi = np.cos(delta_phi)

    target = normalize_target(
        x_left=x_left,
        x_right=x_right,
        sigma_left=sigma_left,
        sigma_right=sigma_right,
        A_left=A_left,
        A_right=A_right,
        cos_delta_phi=cos_delta_phi,
    )

    return (
        intensity.astype(np.float32),
        target.astype(np.float32)
    )


def generate_dataset(
    num_samples,
    x_vals,
    batch_size,
    seed=42
):
    """
    Return a generator function that yields exactly num_samples samples.

    The final batch may contain fewer than batch_size samples.
    """

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    def gen():

        rng = np.random.default_rng(seed)

        remaining = num_samples

        while remaining > 0:

            current_batch_size = min(
                batch_size,
                remaining
            )

            X_batch = np.empty(
                (current_batch_size, len(x_vals)),
                dtype=np.float32
            )

            y_batch = np.empty(
                (current_batch_size, 7),
                dtype=np.float32
            )

            for i in range(current_batch_size):

                intensity, target = generate_sample(
                    x_vals,
                    rng
                )

                X_batch[i] = intensity
                y_batch[i] = target

            remaining -= current_batch_size

            yield X_batch, y_batch

    return gen


# ============================================================
# 5. Inverse Neural Network
# ============================================================

def build_inverse_model(
    input_dim=NUM_POINTS,
    output_dim=7
):
    """
    Fully connected inverse regression network.
    """

    model = tf.keras.Sequential([

        layers.Input(
            shape=(input_dim,)
        ),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.20),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.20),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.10),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dense(
            output_dim,
            activation="linear"
        ),
    ])

    return model


# ============================================================
# 6. Main Training Configuration
# ============================================================

x_vals = np.linspace(
    X_MIN,
    X_MAX,
    NUM_POINTS
)

batch_size = 64

train_samples = 200_000
val_samples = 10_000
test_samples = 10_000

# Number of training batches used in each epoch.
#
# 500 batches × 64 samples = 32,000 generated samples/epoch.
#
# The training generator is repeated indefinitely, so each epoch
# receives fresh synthetic samples.
steps_per_epoch = 500

epochs = 100


# ============================================================
# 7. Validation / Test Sets
# ============================================================

print("Generating validation set...")

val_gen = generate_dataset(
    num_samples=val_samples,
    x_vals=x_vals,
    batch_size=batch_size,
    seed=123
)

X_val_batches = []
y_val_batches = []

for X_batch, y_batch in val_gen():
    X_val_batches.append(X_batch)
    y_val_batches.append(y_batch)

X_val = np.concatenate(
    X_val_batches,
    axis=0
)

y_val = np.concatenate(
    y_val_batches,
    axis=0
)


print(
    f"Validation set: "
    f"X={X_val.shape}, "
    f"y={y_val.shape}"
)


print("Generating test set...")

test_gen = generate_dataset(
    num_samples=test_samples,
    x_vals=x_vals,
    batch_size=batch_size,
    seed=456
)

X_test_batches = []
y_test_batches = []

for X_batch, y_batch in test_gen():
    X_test_batches.append(X_batch)
    y_test_batches.append(y_batch)

X_test = np.concatenate(
    X_test_batches,
    axis=0
)

y_test = np.concatenate(
    y_test_batches,
    axis=0
)


print(
    f"Test set: "
    f"X={X_test.shape}, "
    f"y={y_test.shape}"
)


# ============================================================
# 8. Training Dataset
# ============================================================

train_gen = generate_dataset(
    num_samples=train_samples,
    x_vals=x_vals,
    batch_size=batch_size,
    seed=42
)

train_ds = tf.data.Dataset.from_generator(

    train_gen,

    output_signature=(

        tf.TensorSpec(
            shape=(None, NUM_POINTS),
            dtype=tf.float32
        ),

        tf.TensorSpec(
            shape=(None, 7),
            dtype=tf.float32
        ),

    ),

).repeat().prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 9. Build and Compile Model
# ============================================================

model = build_inverse_model()

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="mse",

    metrics=["mae"]
)


model.summary()


# ============================================================
# 10. Callbacks
# ============================================================

callbacks_list = [

    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),

    callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_loss"
    ),

]


# ============================================================
# 11. Training
# ============================================================

print("\nStarting training...\n")

history = model.fit(

    train_ds,

    epochs=epochs,

    steps_per_epoch=steps_per_epoch,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=callbacks_list,

    verbose=1
)


# ============================================================
# 12. Evaluation
# ============================================================

print("\nEvaluating model...\n")

y_pred_norm = model.predict(
    X_test,
    verbose=1
)

# Convert both ground truth and predictions back to
# physical/model parameter units.
y_test_phys = denormalize_targets(
    y_test
)

y_pred_phys = denormalize_targets(
    y_pred_norm
)


# ------------------------------------------------------------
# Parameter-wise MAE
# ------------------------------------------------------------

parameter_names = [

    "x_left",
    "x_right",
    "sigma_left",
    "sigma_right",
    "A_left",
    "A_right",
    "cos_delta_phi",

]


print("\nPer-parameter MAE:")
print("-" * 50)

for i, name in enumerate(parameter_names):

    mae = np.mean(
        np.abs(
            y_test_phys[:, i]
            - y_pred_phys[:, i]
        )
    )

    print(
        f"{name:18s}: {mae:.6f}"
    )


# ------------------------------------------------------------
# Effective relative phase
# ------------------------------------------------------------

# The current model identifies cos(delta_phi), not the sign
# of delta_phi. Therefore arccos gives the identifiable phase
# representative in [0, pi].

true_cos = np.clip(
    y_test_phys[:, 6],
    -1.0,
    1.0
)

pred_cos = np.clip(
    y_pred_phys[:, 6],
    -1.0,
    1.0
)

phi_true_effective = np.arccos(
    true_cos
)

phi_pred_effective = np.arccos(
    pred_cos
)

phase_mae = np.mean(
    np.abs(
        phi_true_effective
        - phi_pred_effective
    )
)

print(
    f"{'effective_phase':18s}: "
    f"{phase_mae:.6f} rad"
)


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

r2 = r2_score(
    y_test,
    y_pred_norm,
    multioutput="variance_weighted"
)

print(
    f"\nNormalized-target R²: {r2:.6f}"
)


# ============================================================
# 13. Physics-based Reconstruction Error
# ============================================================

print(
    "\nEvaluating reconstructed intensity..."
)

reconstruction_mse = []

for i in range(len(X_test)):

    true_params = y_test_phys[i]
    pred_params = y_pred_phys[i]

    # Ground-truth intensity is already available.
    true_intensity = X_test[i]

    predicted_intensity = (
        compute_intensity_from_cos_phase(
            x_vals=x_vals,
            x_left=pred_params[0],
            x_right=pred_params[1],
            sigma_left=pred_params[2],
            sigma_right=pred_params[3],
            A_left=pred_params[4],
            A_right=pred_params[5],
            cos_delta_phi=np.clip(
                pred_params[6],
                -1.0,
                1.0
            )
        )
    )

    mse_value = np.mean(
        (
            true_intensity
            - predicted_intensity
        ) ** 2
    )

    reconstruction_mse.append(
        mse_value
    )


reconstruction_mse = np.asarray(
    reconstruction_mse
)


print(
    "Mean intensity reconstruction MSE:",
    f"{np.mean(reconstruction_mse):.6e}"
)

print(
    "Median intensity reconstruction MSE:",
    f"{np.median(reconstruction_mse):.6e}"
)


# ============================================================
# 14. Training History Plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="train_loss"
)

plt.plot(
    history.history["val_loss"],
    label="val_loss"
)

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("MSE (normalized targets)")

plt.title(
    "Training History"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "loss_curve.png",
    dpi=150
)

plt.show()


# ============================================================
# 15. Prediction / Reconstruction Visualization
# ============================================================

rng_plot = np.random.default_rng(2026)

num_examples = min(
    5,
    len(X_test)
)

indices = rng_plot.choice(
    len(X_test),
    size=num_examples,
    replace=False
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

axes = axes.flatten()


for plot_index, sample_index in enumerate(indices):

    ax = axes[plot_index]

    # Ground truth intensity
    ax.plot(
        x_vals,
        X_test[sample_index],
        label="True intensity"
    )

    # Predicted parameters
    pred = y_pred_phys[sample_index]

    predicted_intensity = (
        compute_intensity_from_cos_phase(
            x_vals=x_vals,
            x_left=pred[0],
            x_right=pred[1],
            sigma_left=pred[2],
            sigma_right=pred[3],
            A_left=pred[4],
            A_right=pred[5],
            cos_delta_phi=np.clip(
                pred[6],
                -1.0,
                1.0
            )
        )
    )

    ax.plot(
        x_vals,
        predicted_intensity,
        "--",
        label="Predicted reconstruction"
    )

    ax.set_title(
        f"Sample {sample_index}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Intensity")

    ax.legend(fontsize=8)

    ax.grid(True)


# Hide unused subplot(s)
for extra_ax in axes[num_examples:]:
    extra_ax.axis("off")


plt.tight_layout()

plt.savefig(
    "predictions.png",
    dpi=150
)

plt.show()


# ============================================================
# 16. Save Final Model
# ============================================================

model.save(
    "inverse_double_slit_final.keras"
)

print(
    "\nDone."
)

print(
    "Final model saved as: "
    "inverse_double_slit_final.keras"
)
