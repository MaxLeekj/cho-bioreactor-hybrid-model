import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Bioreactor ODEs based on Jang et al. model
def bioreactor_jang(t, y):
    # Unpack variables ensuring values are never negative
    C_X, C_glc, C_gln, C_lac, C_amm, C_Ab = (max(1e-3, var) for var in y)

    # Unpack parameters
    (
        mu_max,
        mud_max,
        Kglc,
        Kgln,
        KIlac,
        KIamm,
        Kdamm,
        Qc,
        Qg0,
        Yxglc,
        mglc,
        Yxgln,
        Qexglc,
        Kexglc,
        Ylacglc,
        Yammgln,
    ) = (
        0.065,
        0.075,
        0.75,
        0.075,
        90,
        15,
        4.5,
        0.7e-9,
        1e-9,
        2.37 * 10**8,
        2 * 10**-12,
        8 * 10**8,
        2 * 10**-10,
        10,
        2,
        0.7,
    )  # known

    a, b, Kdgln, n = [1.87529661, 2.28108721, 0.02590942, 0.01144511]
    # a, b, Kdgln, n = [2, 2.5, 0.02, 0.1]
    # a, b, Kdgln, n = [2, 3, 0.1, 3]

    # No growth when death phase begins
    mu = (
        mu_max
        * C_glc
        / (Kglc + C_glc)
        * C_gln
        / (Kgln + C_gln)
        * KIlac
        / (KIlac + C_lac)
        * KIamm
        / (KIamm + C_amm)
    )
    mu_d = mud_max / (1 + (Kdamm / C_amm) ** n)

    # Define auxiliary terms for inhibition when near threshold
    Qglc = mu / Yxglc + mglc + Qexglc * (C_glc / (C_glc + Kexglc))
    Qgln = mu / Yxgln
    Qlac = Ylacglc * Qglc
    Qamm = Yammgln * Qgln
    fg0 = (mu_d - a) / b
    Qab = Qc * (1 - fg0) + Qg0 * fg0

    # System of ODEs
    dC_X_dt = (mu * (1 - fg0) - mu_d) * C_X
    dC_glc_dt = -Qglc * C_X
    dC_gln_dt = -Qgln * C_X - Kdgln * C_gln
    dC_lac_dt = Qlac * C_X
    dC_amm_dt = Qamm * C_X + Kdgln * C_gln
    dC_Ab_dt = Qab * (C_gln / (C_gln + Kgln)) * C_X

    return np.array([dC_X_dt, dC_glc_dt, dC_gln_dt, dC_lac_dt, dC_amm_dt, dC_Ab_dt])


def bioreactor_odes(t, y, Fin=0):
    # Unpack variables
    C_X, C_glc, C_lac, C_Ab, V = y

    # Enforce minimum volume
    V = max(1e-6, V)  # ensure volume never 0

    # Parameters
    (KIlac, mglc, Qp, mu_max, kd, Yxglc, Ylacglc, Kdglc, cin) = (
        7.1,
        82.3e-12,
        8.18e-13,
        5.17e-2,
        2.32e-2,
        1.33e9,
        2.76e-11,
        1.54,
        722,
    )

    # Pre-compute safe Fin/V
    Fin_over_V = Fin / V

    # Growth and death rates
    mu = mu_max * (KIlac / (KIlac + max(0, C_lac)))
    mu_d = kd * (Kdglc / (Kdglc + max(0, C_glc)))

    # Calculate derivatives with non-negativity constraints
    # Cell concentration
    dC_X_dt = (mu - mu_d) * max(0, C_X) - Fin_over_V * max(0, C_X)
    if C_X <= 0 and dC_X_dt < 0:
        dC_X_dt = 0

    # Glucose concentration
    dC_glc_dt = -((mu - mu_d) / Yxglc + mglc) * max(0, C_X) + Fin_over_V * (
        cin - max(0, C_glc)
    )
    if C_glc <= 0 and dC_glc_dt < 0:
        dC_glc_dt = 0

    # Lactate concentration
    dC_lac_dt = Ylacglc * max(0, C_X) - Fin_over_V * max(0, C_lac)
    if C_lac <= 0 and dC_lac_dt < 0:
        dC_lac_dt = 0

    # Antibody concentration
    dC_Ab_dt = Qp * max(0, C_X) - Fin_over_V * max(0, C_Ab)
    if C_Ab <= 0 and dC_Ab_dt < 0:
        dC_Ab_dt = 0

    # Volume change
    dV_dt = Fin

    return [dC_X_dt, dC_glc_dt, dC_lac_dt, dC_Ab_dt, dV_dt]


# # Fin as a function of time
# def get_Fin(t, V):
#     if t < 60:
#         vvd = 0
#     elif 60 <= t < 150:
#         vvd = 0.05
#     elif 150 <= t < 220:
#         vvd = 0.06
#     else:  # t >= 220
#         vvd = 0.07
#     return vvd * V / 24  # Compute Fin


def bioreactor_balanced(t, y, S, bioreactor):
    v = bioreactor(t, y)
    dXdt = S @ v
    return dXdt


def plot_model_vs_ode(
    model,
    scaler_X,
    scaler_Y,
    bioreactor_odes,
    initial_conditions,
    Fin,
    t_span,
    t_eval,
    output_cols,
):
    """
    Compare model predictions vs ODE solver outputs.

    Parameters:
        model             : trained Neural Network model
        scaler_X          : fitted input scaler
        scaler_Y          : fitted output scaler
        bioreactor_odes   : ODE function (t, y, Fin) -> dy/dt
        initial_conditions: list of 5 states [C_X0, C_glc0, C_lac0, C_Ab0, V0]
        Fin               : feed rate (float)
        t_span            : tuple (t0, tf)
        t_eval            : array of time points
        output_cols       : list of state names
    """
    num_states = len(output_cols)
    time_steps = len(t_eval)

    # === 1. Run ODE solver ===
    y0 = initial_conditions  # just the states
    sol = solve_ivp(
        lambda t, y: bioreactor_odes(t, y, Fin),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    Y_ode = sol.y.T  # (time_steps, num_states)

    # === 2. Prepare Neural Network input ===
    X_input = np.array(initial_conditions + [Fin]).reshape(1, -1)  # (1, num_inputs)
    X_scaled = scaler_X.transform(X_input)

    # === 3. Run Neural Network prediction ===
    Y_pred_scaled = model.predict(X_scaled)  # (1, time_steps, num_states)
    Y_pred_scaled = Y_pred_scaled.reshape(-1, num_states)

    # Inverse scaling
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled).reshape(time_steps, num_states)

    # === 4. Plot results ===
    fig, axs = plt.subplots(num_states, 1, figsize=(8, 2 * num_states), sharex=True)
    for i, col in enumerate(output_cols):
        axs[i].plot(t_eval, Y_ode[:, i], "k-", label="ODE")
        axs[i].plot(t_eval, Y_pred[:, i], "r--", label="LSTM")
        axs[i].set_ylabel(col)
        axs[i].legend()

    axs[-1].set_xlabel("Time")
    plt.suptitle("Model vs ODE Trajectories")
    plt.tight_layout()
    plt.show()

    return Y_ode, Y_pred


def plot_worst_predictions(model, X, Y_true, num_worst=3):
    """
    Find and plot the worst predictions (highest MSE) after training.

    Args:
        model: trained keras model
        X: input data
        Y_true: ground truth sequences (shape: [n_samples, time_steps, num_states])
        num_worst: number of worst predictions to visualize
    """
    # Get predictions
    Y_pred = model.predict(X, verbose=0)

    # Compute MSE per sample
    errors = np.mean((Y_true - Y_pred) ** 2, axis=(1, 2))  # average over time + states

    # Get indices of worst samples
    worst_idx = np.argsort(errors)[-num_worst:][::-1]  # descending

    for rank, i in enumerate(worst_idx, 1):
        plt.figure(figsize=(12, 6))
        for state in range(Y_true.shape[2]):  # loop over states
            plt.subplot(Y_true.shape[2], 1, state + 1)
            plt.plot(Y_true[i, :, state], label="True", color="black")
            plt.plot(Y_pred[i, :, state], label="Pred", linestyle="--")
            if state == 0:
                plt.title(f"Worst Prediction #{rank} (sample {i}, MSE={errors[i]:.4e})")
            plt.ylabel(f"State {state+1}")
            plt.ylim(bottom=0)  # ensure y-axis starts at 0
            plt.legend()
        plt.xlabel("Time step")
        plt.tight_layout()
        plt.show()


# ===============================================
# Generic NN Save / Load / Predict (min–max)
# ===============================================
import os, json, numpy as np, keras


# ---------- SAVE ----------
def save_nn_artifacts(
    model,
    save_dir,
    *,
    x_min,
    x_rng,
    y_min,
    y_rng,
    F_time,
    t_eval,
    y0_names=None,
    extra_meta=None,
):
    """
    Save a trained Keras model + normalization + time features.

    Args:
      model      : Keras model (any architecture with input (N,T,D), output (N,T,1))
      save_dir   : folder to write artifacts
      x_min/x_rng: (D_y0,) min & range for input min–max scaling (y0 features)
      y_min/y_rng: (1,1,1) min & range for target min–max (Xv(t))
      F_time     : (T, Fd) time features used at training
      t_eval     : (T,) time grid (hours, or whatever you used)
      y0_names   : optional list of names for y0 features (e.g. ["Xv","GLC","GLN","LAC","NH3","Ab"])
      extra_meta : optional dict of notes (hyperparams, git hash, dataset tag, etc.)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Model (Keras v3 format)
    model.save(os.path.join(save_dir, "model.keras"))

    # 2) Scalers
    np.savez_compressed(
        os.path.join(save_dir, "scalers.npz"),
        x_min=np.asarray(x_min, np.float32),
        x_rng=np.asarray(x_rng, np.float32),
        y_min=np.asarray(y_min, np.float32),
        y_rng=np.asarray(y_rng, np.float32),
    )

    # 3) Time features and grid
    np.save(os.path.join(save_dir, "F_time.npy"), np.asarray(F_time, np.float32))
    np.save(os.path.join(save_dir, "t_eval.npy"), np.asarray(t_eval, np.float32))

    # 4) Meta
    meta = {
        "keras_version": keras.__version__,
        "scaler_mode": "minmax",  # change if you ever use zscore, etc.
        "T": int(np.asarray(t_eval).shape[0]),
        "F_dim": int(np.asarray(F_time).shape[1]),
        "y0_dim": int(np.asarray(x_min).shape[0]),
        "y0_names": y0_names or [],
        "notes": extra_meta or {},
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] Model + artifacts saved to: {save_dir}")


# ---------- LOAD ----------
def load_nn_artifacts(save_dir):
    """
    Load everything needed for inference.
    Returns dict: {model, x_min, x_rng, y_min, y_rng, F_time, t_eval, meta}
    """
    model = keras.models.load_model(os.path.join(save_dir, "model.keras"))
    sc = np.load(os.path.join(save_dir, "scalers.npz"))
    F_time = np.load(os.path.join(save_dir, "F_time.npy"))
    t_eval = np.load(os.path.join(save_dir, "t_eval.npy"))
    try:
        with open(os.path.join(save_dir, "meta.json"), "r") as f:
            meta = json.load(f)
    except Exception:
        meta = {"scaler_mode": "minmax"}

    return {
        "model": model,
        "x_min": sc["x_min"].astype(np.float32),
        "x_rng": sc["x_rng"].astype(np.float32),
        "y_min": sc["y_min"].astype(np.float32),
        "y_rng": sc["y_rng"].astype(np.float32),
        "F_time": F_time.astype(np.float32),
        "t_eval": t_eval.astype(np.float32),
        "meta": meta,
    }


# ---------- BUILD INPUT (from one y0) ----------
def build_seq_input_from_y0(art, y0_user):
    """
    Build a single (1, T, D) input by concatenating normalized y0 with saved F_time.
    y0_user : [Xv, GLC, GLN, LAC, NH3, Ab] (units as in your training)
    """
    x_min, x_rng = art["x_min"], art["x_rng"]
    F_time = art["F_time"]
    T = F_time.shape[0]

    y0 = np.array(y0_user, np.float32).reshape(-1)
    if y0.shape[0] != x_min.shape[0]:
        raise ValueError(f"y0 has dim {y0.shape[0]} but expected {x_min.shape[0]}.")

    Xn = ((y0 - x_min) / x_rng)[None, :]  # (1, D_y0) in [0,1]
    Xrep = np.repeat(Xn[:, None, :], T, axis=1)  # (1, T, D_y0)
    X_in = np.concatenate([Xrep, F_time[None, :, :]], axis=2).astype(
        np.float32
    )  # (1, T, D)
    return X_in


# ---------- DENORMALIZE ----------
def denorm_targets(art, Yn):
    """
    Yn: (N, T, 1) in [0,1]  ->  raw cells/L via saved y_min/y_rng
    """
    y_min, y_rng = art["y_min"], art["y_rng"]
    return (Yn.astype(np.float32) * y_rng + y_min).astype(np.float32)


# ---------- PREDICT (one y0) ----------
def predict_Xv_from_y0_saved(art, y0_user):
    """
    Returns t_eval (T,), Xv_hat (T,) in cells/L.
    """
    model = art["model"]
    t_eval = art["t_eval"]

    X_in = build_seq_input_from_y0(art, y0_user)  # (1,T,D)
    Yn = model.predict(X_in, verbose=0)[0]  # (T,1) in [0,1]
    Y = denorm_targets(art, Yn[None, ...])[0, :, 0]  # (T,)
    return t_eval, Y


# ---------- PREDICT (batch of y0) ----------
def predict_batch_from_y0_saved(art, Y0_batch):
    """
    Y0_batch: (N, D_y0)
    Returns: t_eval (T,), Xv_hat (N, T) in cells/L
    """
    model = art["model"]
    t_eval = art["t_eval"]
    F_time = art["F_time"]
    T = F_time.shape[0]

    Y0_batch = np.asarray(Y0_batch, np.float32)
    if Y0_batch.ndim != 2 or Y0_batch.shape[1] != art["x_min"].shape[0]:
        raise ValueError(
            f"Y0_batch must be (N,{art['x_min'].shape[0]}). Got {Y0_batch.shape}."
        )

    # Normalize & tile
    Xn = ((Y0_batch - art["x_min"]) / art["x_rng"]).astype(np.float32)  # (N, D_y0)
    Xrep = np.repeat(Xn[:, None, :], T, axis=1)  # (N, T, D_y0)
    Xin = np.concatenate(
        [Xrep, np.repeat(F_time[None, :, :], Y0_batch.shape[0], axis=0)], axis=2
    )

    Yn = model.predict(Xin, verbose=0)  # (N, T, 1)
    Y = denorm_targets(art, Yn)[:, :, 0]  # (N, T)
    return t_eval, Y


# ---------- OPTIONAL: Compare with ODE truth ----------
def compare_model_vs_ode_saved(
    art, y0_user, simulate_traj_fn, title=None, return_data=False
):
    """
    simulate_traj_fn: function(y0, t_eval) -> (T,6) full state (uses your utils.bioreactor_jang internally)
    """
    import matplotlib.pyplot as plt

    t_eval, Xv_hat = predict_Xv_from_y0_saved(art, y0_user)
    Ytruth = simulate_traj_fn(np.array(y0_user, np.float32), t_eval)
    Xv_true = Ytruth[:, 0].astype(np.float32)

    rmse = float(np.sqrt(np.mean((Xv_true - Xv_hat) ** 2)))
    ss_res = np.sum((Xv_true - Xv_hat) ** 2)
    ss_tot = np.sum((Xv_true - Xv_true.mean()) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    plt.figure()
    plt.plot(t_eval, Xv_true, label="Truth (ODE)", lw=2)
    plt.plot(t_eval, Xv_hat, "--", label="NN")
    ttl = title or "Model vs ODE — Xv(t)"
    plt.title(f"{ttl}\nRMSE={rmse:.3e} cells/L | R²={r2:.4f}")
    plt.xlabel("Time [h]")
    plt.ylabel("Xv [cells/L]")
    plt.legend()
    plt.show()
    plt.close()

    if return_data:
        return {
            "t": t_eval,
            "Xv_true": Xv_true,
            "Xv_pred": Xv_hat,
            "rmse": rmse,
            "r2": r2,
        }
