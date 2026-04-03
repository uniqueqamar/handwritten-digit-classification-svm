
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_prepare_data(test_size: float = 0.3, seed: int = RANDOM_SEED):
   
    
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Fit scaler only on training data to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Recover 8×8 images for the test split (for visual analysis)
    # digits.images is indexed in original order; we need the test indices
    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    images_test = digits.images[test_idx]

    return X_train_scaled, X_test_scaled, X_test, y_train, y_test, images_test, scaler


# =============================================================================
# 2. MODEL TRAINING
# =============================================================================

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:

 
    model = SVC(kernel="rbf", C=10, gamma=0.001, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 3. NOISE INJECTION
# =============================================================================

def inject_gaussian_noise(X: np.ndarray, noise_std: float, seed: int = RANDOM_SEED) -> np.ndarray:
 
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    return X + noise


# =============================================================================
# 4. EVALUATION PIPELINE
# =============================================================================

def evaluate_across_noise_levels(
    model: SVC,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list,
) -> dict:
    
    results = {
        "noise_levels": noise_levels,
        "accuracies": [],
        "per_class_f1": [],   # shape: (n_noise_levels, 10)
        "confusion_matrices": [],
    }

    for std in noise_levels:
        X_noisy = inject_gaussian_noise(X_test_scaled, noise_std=std)
        y_pred = model.predict(X_noisy)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_per_class = [report[str(c)]["f1-score"] for c in range(10)]
        cm = confusion_matrix(y_test, y_pred)

        results["accuracies"].append(acc)
        results["per_class_f1"].append(f1_per_class)
        results["confusion_matrices"].append(cm)

    results["per_class_f1"] = np.array(results["per_class_f1"])
    return results


# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

def plot_sample_digits(images_test: np.ndarray, y_test: np.ndarray,
                       n: int = 10, save_path: str = "sample_digits.png"):
    """Show a row of sample test digits so the reader understands the data."""
    fig, axes = plt.subplots(1, n, figsize=(14, 2))
    fig.suptitle("Sample Digits from Test Set", fontsize=13, y=1.05)
    for i, ax in enumerate(axes):
        ax.imshow(images_test[i], cmap="gray_r")
        ax.set_title(str(y_test[i]), fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                          title: str = "Confusion Matrix (Baseline — No Noise)",
                          save_path: str = "confusion_matrix_baseline.png"):
 
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[str(i) for i in range(10)])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_accuracy_vs_noise(results: dict, save_path: str = "accuracy_vs_noise.png"):
  
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["noise_levels"], results["accuracies"],
            marker="o", linewidth=2.5, color="#2563EB", markersize=7)
    ax.fill_between(results["noise_levels"], results["accuracies"],
                    alpha=0.1, color="#2563EB")
    ax.set_xlabel("Gaussian Noise Standard Deviation (σ)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Robustness: Accuracy vs. Input Noise Level", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=results["accuracies"][0], color="gray", linestyle="--",
               alpha=0.7, label=f"Baseline accuracy ({results['accuracies'][0]:.3f})")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_per_class_f1_heatmap(results: dict, save_path: str = "per_class_f1_heatmap.png"):

 
    f1_matrix = results["per_class_f1"]   # shape: (n_levels, 10)
    noise_labels = [f"σ={s:.1f}" for s in results["noise_levels"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(f1_matrix.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {i}" for i in range(10)], fontsize=10)
    ax.set_xlabel("Noise Level", fontsize=11)
    ax.set_title("Per-Class F1 Score Across Noise Levels\n"
                 "(Green = robust, Red = fragile)", fontsize=12)
    plt.colorbar(im, ax=ax, label="F1 Score")

    # Annotate cells with values for readability
    for i in range(len(noise_labels)):
        for j in range(10):
            ax.text(i, j, f"{f1_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if f1_matrix[i, j] > 0.4 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_vulnerability_ranking(results: dict, save_path: str = "digit_vulnerability.png"):
   
    baseline_f1 = results["per_class_f1"][0]       # clean data
    worst_f1    = results["per_class_f1"][-1]       # highest noise
    drop        = baseline_f1 - worst_f1            # larger = more vulnerable

    colours = ["#EF4444" if d > np.median(drop) else "#22C55E" for d in drop]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(10), drop, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"Digit {i}" for i in range(10)])
    ax.set_ylabel("F1 Drop  (baseline − high-noise)", fontsize=11)
    ax.set_title("Digit Vulnerability to Gaussian Noise\n"
                 "(Red bars = above-median vulnerability)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, drop):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_misclassified_samples(
    images_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    noise_std: float,
    n_samples: int = 16,
    save_path: str = "misclassified_samples.png",
):
    wrong_idx = np.where(y_test != y_pred)[0]
    if len(wrong_idx) == 0:
        print("  No misclassifications at this noise level.")
        return

    n_show = min(n_samples, len(wrong_idx))
    cols = 8
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = axes.flatten()
    fig.suptitle(f"Misclassified Samples at σ={noise_std:.1f}  "
                 f"({len(wrong_idx)} total errors)", fontsize=12)

    for i, idx in enumerate(wrong_idx[:n_show]):
        ax = axes[i]
        ax.imshow(images_test[idx], cmap="gray_r")
        ax.set_title(f"T:{y_test[idx]} → P:{y_pred[idx]}", fontsize=8,
                     color="#DC2626")
        ax.axis("off")

    # Hide unused subplots
    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


def plot_noisy_vs_clean(
    X_test_scaled: np.ndarray,
    images_test: np.ndarray,
    scaler: StandardScaler,
    noise_std: float,
    n: int = 5,
    save_path: str = "noisy_vs_clean.png",
):
    
    rng = np.random.default_rng(RANDOM_SEED)
    X_noisy = X_test_scaled + rng.normal(0, noise_std, X_test_scaled.shape)

    # Inverse transform to pixel space for display
    X_clean_px = scaler.inverse_transform(X_test_scaled).reshape(-1, 8, 8)
    X_noisy_px = scaler.inverse_transform(X_noisy).reshape(-1, 8, 8)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4.5))
    fig.suptitle(f"Clean vs. Noisy Digits  (σ={noise_std:.1f})", fontsize=12)

    for i in range(n):
        axes[0, i].imshow(images_test[i], cmap="gray_r")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Clean", fontsize=10)

        axes[1, i].imshow(X_noisy_px[i], cmap="gray_r")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Noisy", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {save_path}")


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("  Robustness Analysis of SVM Digit Classifier")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading and splitting data...")
    (X_train_scaled, X_test_scaled,
     X_test_raw, y_train, y_test,
     images_test, scaler) = load_and_prepare_data()
    print(f"      Train: {X_train_scaled.shape[0]} samples | "
          f"Test: {X_test_scaled.shape[0]} samples")

    # ── Sample visualisation ───────────────────────────────────────
    print("\n[2/6] Visualising sample digits...")
    plot_sample_digits(images_test, y_test)

    # ── Train model ────────────────────────────────────────────────
    print("\n[3/6] Training SVM classifier...")
    model = train_svm(X_train_scaled, y_train)

    # ── Baseline evaluation ────────────────────────────────────────
    print("\n[4/6] Baseline evaluation (clean test data)...")
    y_pred_clean = model.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, y_pred_clean)
    print(f"\n  Baseline Accuracy: {baseline_acc:.4f}\n")
    print(classification_report(y_test, y_pred_clean,
                                target_names=[str(i) for i in range(10)]))
    plot_confusion_matrix(y_test, y_pred_clean)

    # ── Noise robustness evaluation ────────────────────────────────
    print("\n[5/6] Evaluating robustness across noise levels...")
    # Range chosen so σ=0 (clean) through σ=2.0 (heavy noise);
    # 11 points gives a smooth curve without excess compute.
    noise_levels = [round(x, 1) for x in np.linspace(0.0, 2.0, 11)]
    results = evaluate_across_noise_levels(model, X_test_scaled, y_test, noise_levels)

    for lvl, acc in zip(noise_levels, results["accuracies"]):
        print(f"      σ={lvl:.1f}  →  accuracy = {acc:.4f}")

    # ── Visualise degradation ──────────────────────────────────────
    print("\n[6/6] Generating analysis plots...")
    plot_accuracy_vs_noise(results)
    plot_per_class_f1_heatmap(results)
    plot_vulnerability_ranking(results)

    # Misclassified samples at moderate noise (σ=1.0) and high noise (σ=2.0)
    for noise_std in [1.0, 2.0]:
        X_noisy = inject_gaussian_noise(X_test_scaled, noise_std)
        y_pred_noisy = model.predict(X_noisy)
        plot_misclassified_samples(
            images_test, y_test, y_pred_noisy,
            noise_std=noise_std,
            save_path=f"misclassified_sigma_{int(noise_std*10):02d}.png",
        )

    # Clean vs noisy side-by-side at σ=1.0
    plot_noisy_vs_clean(X_test_scaled, images_test, scaler, noise_std=1.0)

    # ── Summary insight ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    f1_drop = results["per_class_f1"][0] - results["per_class_f1"][-1]
    most_vulnerable = int(np.argmax(f1_drop))
    most_robust = int(np.argmin(f1_drop))
    print(f"  Baseline accuracy          : {results['accuracies'][0]:.4f}")
    print(f"  Accuracy at σ=2.0          : {results['accuracies'][-1]:.4f}")
    print(f"  Total accuracy drop        : {results['accuracies'][0] - results['accuracies'][-1]:.4f}")
    print(f"  Most vulnerable digit      : {most_vulnerable}  (F1 drop = {f1_drop[most_vulnerable]:.3f})")
    print(f"  Most robust digit          : {most_robust}  (F1 drop = {f1_drop[most_robust]:.3f})")
    print("=" * 65)
    print("\nAll plots saved. Analysis complete.")


if __name__ == "__main__":
    main()
