import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from datetime import datetime
import os
import pandas as pd
import csv


from src.config import TRAINING_LOG_DIR, CHECKPOINT_DIR


def _get_model_display_name(model):
    """Build a descriptive checkpoint filename from model attributes."""
    cls_name = model.__class__.__name__
    att = getattr(model, 'attention_type', None)
    red = getattr(model, 'reduction_ratio', None)
    if att == 'none':
        return f"{cls_name}_NoAttn"
    elif att == 'cbam' and red:
        return f"{cls_name}_CBAM_r{red}"
    elif att == 'se' and red:
        return f"{cls_name}_SE_r{red}"
    return cls_name


# Define EarlyStopping class with checkpoint saving
class EarlyStopping:
    def __init__(
        self, patience=5, delta=0, save_best_model=True, checkpoint_dir=CHECKPOINT_DIR
    ):
        """
        Args:
        patience (int): How many epochs to wait after the last improvement before stopping.
        delta (float): Minimum change to qualify as an improvement.
        save_best_model (bool): Whether to save the best model based on validation loss.
        checkpoint_dir (str): Directory where to save the best model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.save_best_model = save_best_model
        self.checkpoint_dir = checkpoint_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None
        self.best_checkpoint_path = None
        self.best_epoch = 0

        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        """
        Call the early stopping function to decide whether to stop training.

        Args:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): The model to save the best weights for.
        epoch (int): Current epoch number to include in the checkpoint file name.
        """
        score = -val_loss  # We minimize loss, so we negate it to maximize the score
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if self.save_best_model:
                self.best_model_wts = model.state_dict()  # Save the best weights
                self.save_checkpoint(model, epoch)  # Save the model checkpoint
            self.counter = 0

    def save_checkpoint(self, model, epoch):
        """Save the best model checkpoint, overwriting previous best model."""
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            os.remove(self.best_checkpoint_path)  # Remove previous best checkpoint

        model_name = _get_model_display_name(model)
        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{model_name}_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), self.best_checkpoint_path)
        print(f"Best checkpoint saved to {self.best_checkpoint_path}")

    def load_best_model(self, model):
        """Load the best model's weights."""
        if self.save_best_model and self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
            print(f"Loaded best model weights from epoch {self.best_epoch}")


class TrainingLogger:
    def __init__(
        self,
        model_name=str,
        log_dir=TRAINING_LOG_DIR,
        enable_live_plot=True,
        plot_every=1,
    ):
        """
        Initialize the training logger with enhanced visualization capabilities.

        Args:
            model_name (str): The name of the model being trained
            log_dir (str): Directory to save log files
            enable_live_plot (bool): Whether to show live plots during training
            plot_every (int): Update plots every N epochs
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.model_name = model_name if model_name else "model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, f"{self.model_name}_training_log_{timestamp}.csv"
        )
        self.results = []
        self.enable_live_plot = enable_live_plot
        self.plot_every = plot_every

        # Training visualization setup
        if self.enable_live_plot:
            plt.ion()  # Turn on interactive mode
            self.fig, ((self.ax_loss, self.ax_acc), (self.ax_lr, self.ax_time)) = (
                plt.subplots(2, 2, figsize=(15, 10))
            )
            self.fig.suptitle(
                f"{self.model_name} Training Progress", fontsize=16, fontweight="bold"
            )

            # Initialize plots
            self._setup_plots()

        # Training metrics tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epoch_times = []
        self.learning_rates = []

    def _setup_plots(self):
        """Setup the initial plot configuration."""
        # Loss plot
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title("Training & Validation Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()

        # Accuracy plot
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_title("Training & Validation Accuracy")
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.set_ylim(0, 1)

        # Learning rate plot
        self.ax_lr.set_xlabel("Epoch")
        self.ax_lr.set_ylabel("Learning Rate")
        self.ax_lr.set_title("Learning Rate Schedule")
        self.ax_lr.grid(True, alpha=0.3)
        self.ax_lr.set_yscale("log")

        # Training time plot
        self.ax_time.set_xlabel("Epoch")
        self.ax_time.set_ylabel("Time per Epoch (s)")
        self.ax_time.set_title("Training Speed")
        self.ax_time.grid(True, alpha=0.3)

        plt.tight_layout()

    def log_epoch(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        epoch_time=None,
        learning_rate=None,
    ):
        """Log results for one epoch with enhanced tracking."""
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "learning_rate": learning_rate,
        }
        self.results.append(epoch_data)

        # Track best performance
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Store metrics for plotting
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)

        # Update live plots
        if self.enable_live_plot and epoch % self.plot_every == 0:
            self._update_live_plots()

    def _update_live_plots(self):
        """Update the live training plots."""
        if not self.results:
            return

        epochs = [r["epoch"] for r in self.results]
        train_losses = [r["train_loss"] for r in self.results]
        val_losses = [r["val_loss"] for r in self.results]
        train_accs = [r["train_acc"] for r in self.results]
        val_accs = [r["val_acc"] for r in self.results]

        # Clear and update loss plot
        self.ax_loss.clear()
        self.ax_loss.plot(
            epochs, train_losses, "b-", label="Training Loss", linewidth=2
        )
        self.ax_loss.plot(
            epochs, val_losses, "r-", label="Validation Loss", linewidth=2
        )
        # Highlight best validation loss
        best_val_idx = val_losses.index(min(val_losses))
        self.ax_loss.plot(
            epochs[best_val_idx],
            val_losses[best_val_idx],
            "ro",
            markersize=8,
            label=f"Best Val Loss (Epoch {epochs[best_val_idx]})",
        )
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title("Training & Validation Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend()

        # Clear and update accuracy plot
        self.ax_acc.clear()
        self.ax_acc.plot(
            epochs, train_accs, "b-", label="Training Accuracy", linewidth=2
        )
        self.ax_acc.plot(
            epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2
        )
        # Highlight best validation accuracy
        best_acc_idx = val_accs.index(max(val_accs))
        self.ax_acc.plot(
            epochs[best_acc_idx],
            val_accs[best_acc_idx],
            "ro",
            markersize=8,
            label=f"Best Val Acc (Epoch {epochs[best_acc_idx]})",
        )
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_title("Training & Validation Accuracy")
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.legend()

        # Update learning rate plot
        if self.learning_rates:
            self.ax_lr.clear()
            self.ax_lr.plot(
                epochs[: len(self.learning_rates)],
                self.learning_rates,
                "g-",
                linewidth=2,
            )
            self.ax_lr.set_xlabel("Epoch")
            self.ax_lr.set_ylabel("Learning Rate")
            self.ax_lr.set_title("Learning Rate Schedule")
            self.ax_lr.grid(True, alpha=0.3)
            self.ax_lr.set_yscale("log")

        # Update training time plot
        if self.epoch_times:
            self.ax_time.clear()
            self.ax_time.plot(
                epochs[: len(self.epoch_times)], self.epoch_times, "m-", linewidth=2
            )
            self.ax_time.set_xlabel("Epoch")
            self.ax_time.set_ylabel("Time per Epoch (s)")
            self.ax_time.set_title("Training Speed")
            self.ax_time.grid(True, alpha=0.3)

            # Add average time annotation
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            self.ax_time.axhline(
                y=avg_time,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Avg: {avg_time:.2f}s",
            )
            self.ax_time.legend()

        plt.tight_layout()
        plt.pause(0.1)  # Small pause to allow plot update

    def save_to_csv(self):
        """Save all results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.log_file, index=False)
        print(f"Training log saved to {self.log_file}")

    def save_final_plots(self):
        """Save final training plots to files."""
        if not self.enable_live_plot or not self.results:
            return

        # Save the current plot
        plot_path = os.path.join(
            self.log_dir, f"{self.model_name}_training_progress.png"
        )
        self.fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training plots saved to {plot_path}")

        # Create a summary text file
        summary_path = os.path.join(
            self.log_dir, f"{self.model_name}_training_summary.txt"
        )
        with open(summary_path, "w") as f:
            f.write(f"Training Summary for {self.model_name}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Epochs: {len(self.results)}\n")
            f.write(
                f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})\n"
            )
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n")
            if self.epoch_times:
                f.write(
                    f"Average Time per Epoch: {sum(self.epoch_times) / len(self.epoch_times):.2f}s\n"
                )
                f.write(f"Total Training Time: {sum(self.epoch_times):.2f}s\n")
        print(f"Training summary saved to {summary_path}")

        # Close the plot to free memory
        plt.ioff()
        plt.close(self.fig)


def quick_plot_metrics(csv_path, save_dir=None):
    """
    Quickly plot training metrics from a saved CSV file.

    Args:
        csv_path: Path to the training log CSV file
        save_dir: Optional directory to save the plots
    """
    try:
        df = pd.read_csv(csv_path)

        # Convert to the expected format
        history = {
            "train_loss": df["train_loss"].tolist(),
            "val_loss": df["val_loss"].tolist(),
            "train_accuracy": df["train_acc"].tolist(),
            "val_accuracy": df["val_acc"].tolist(),
        }

        model_name = os.path.basename(csv_path).split("_")[0]
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_training_history.png")

        plot_training_history(history, save_path, model_name)

    except Exception as e:
        print(f"Error plotting metrics from CSV: {e}")


def compare_training_runs(csv_paths, model_names=None, save_path=None):
    """
    Compare multiple training runs on the same plot.

    Args:
        csv_paths: List of paths to training log CSV files
        model_names: Optional list of model names for labels
        save_path: Optional path to save the comparison plot
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(csv_paths))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Training Runs Comparison", fontsize=16, fontweight="bold")

    colors = plt.cm.tab10(range(len(csv_paths)))

    for i, (csv_path, model_name) in enumerate(zip(csv_paths, model_names)):
        try:
            df = pd.read_csv(csv_path)
            epochs = range(1, len(df) + 1)
            color = colors[i]

            # Plot losses
            ax1.plot(
                epochs,
                df["train_loss"],
                color=color,
                linestyle="-",
                label=f"{model_name} (Train)",
                alpha=0.7,
            )
            ax1.plot(
                epochs,
                df["val_loss"],
                color=color,
                linestyle="--",
                label=f"{model_name} (Val)",
                alpha=0.9,
            )

            # Plot accuracies
            ax2.plot(
                epochs,
                df["train_acc"],
                color=color,
                linestyle="-",
                label=f"{model_name} (Train)",
                alpha=0.7,
            )
            ax2.plot(
                epochs,
                df["val_acc"],
                color=color,
                linestyle="--",
                label=f"{model_name} (Val)",
                alpha=0.9,
            )

        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    # Customize loss plot
    ax1.set_xlabel("Epochs", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Loss Comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Customize accuracy plot
    ax2.set_xlabel("Epochs", fontweight="bold")
    ax2.set_ylabel("Accuracy", fontweight="bold")
    ax2.set_title("Accuracy Comparison", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def training_dashboard_summary(log_dir="results/training"):
    """
    Create a dashboard summary of all training runs in the log directory.

    Args:
        log_dir: Directory containing training log CSV files
    """
    csv_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return

    print(f"* Training Dashboard Summary")
    print(f"Found {len(csv_files)} training runs:")
    print("=" * 80)

    summary_data = []

    for csv_file in csv_files:
        csv_path = os.path.join(log_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)

            summary = {
                "Model": csv_file.replace("_training_log_", " ").replace(".csv", ""),
                "Epochs": len(df),
                "Best Val Acc": f"{df['val_acc'].max():.4f}",
                "Best Val Loss": f"{df['val_loss'].min():.4f}",
                "Final Val Acc": f"{df['val_acc'].iloc[-1]:.4f}",
                "Final Val Loss": f"{df['val_loss'].iloc[-1]:.4f}",
            }

            if "epoch_time" in df.columns:
                summary["Avg Time/Epoch"] = f"{df['epoch_time'].mean():.2f}s"

            summary_data.append(summary)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(log_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📁 Summary saved to {summary_path}")

    return summary_df


def evaluate_multiple_models(
    model_configs, test_loader, class_names, device, results_dir="results"
):
    """
    Evaluate multiple models and create comparison reports.

    Args:
        model_configs: List of dictionaries with 'name', 'model', 'checkpoint_path'
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run evaluation on
        results_dir: Directory to save results
    """
    import pandas as pd
    import numpy as np

    comparison_results = []
    all_predictions = {}

    print(f"🔬 Evaluating {len(model_configs)} models...")

    for config in model_configs:
        model_name = config["name"]
        model = config["model"]
        checkpoint_path = config.get("checkpoint_path")

        print(f"\n* Evaluating {model_name}...")

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                if "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                else:
                    model.load_state_dict(state_dict)
                print(f"✅ Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")

        # Evaluate model
        criterion = torch.nn.CrossEntropyLoss()
        cm, metrics, final_loss = evaluate_model(
            model,
            criterion,
            test_loader,
            class_names,
            device,
            results_dir=os.path.join(results_dir, f"{model_name}_evaluation"),
        )

        # Store results for comparison
        result_summary = {
            "Model": model_name,
            "Accuracy": metrics["overall"]["Accuracy"],
            "Precision": metrics["overall"]["Precision"],
            "Recall": metrics["overall"]["Recall"],
            "F1 Score": metrics["overall"]["F1 Score"],
            "ROC AUC": metrics["overall"]["ROC AUC"],
            "Final Loss": final_loss,
        }
        comparison_results.append(result_summary)

        # Store predictions for ensemble analysis
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_predictions[model_name] = {
            "predictions": np.array(all_preds),
            "probabilities": np.array(all_probs),
            "labels": np.array(all_labels),
        }

    # Create comparison visualizations
    create_model_comparison_dashboard(
        comparison_results, all_predictions, class_names, results_dir
    )

    return comparison_results, all_predictions


def create_model_comparison_dashboard(
    comparison_results, all_predictions, class_names, results_dir
):
    """
    Create comprehensive model comparison dashboard.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create comparison directory
    comp_dir = os.path.join(results_dir, "model_comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(comparison_results)

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle("Model Comparison Dashboard", fontsize=20, fontweight="bold")

    # 1. Overall Performance Bar Chart
    ax1 = fig.add_subplot(gs[0, 0:2])
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    x = np.arange(len(df))
    width = 0.15

    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i * width, df[metric], width, label=metric, alpha=0.8)

    ax1.set_xlabel("Models")
    ax1.set_ylabel("Score")
    ax1.set_title("Overall Performance Comparison", fontweight="bold")
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 2. Performance Radar Chart
    ax2 = fig.add_subplot(gs[0, 2:4], projection="polar")

    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for idx, row in df.iterrows():
        values = [row[metric] for metric in metrics_to_plot]
        values += values[:1]  # Complete the circle

        ax2.plot(angles, values, "o-", linewidth=2, label=row["Model"])
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_to_plot)
    ax2.set_ylim(0, 1)
    ax2.set_title("Performance Radar Chart", fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    ax2.grid(True)

    # 3. Loss Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(df["Model"], df["Final Loss"], color="coral", alpha=0.7)
    ax3.set_title("Final Loss Comparison", fontweight="bold")
    ax3.set_ylabel("Loss")
    ax3.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # 4. Accuracy Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.boxplot(
        [
            pred_data["predictions"] == pred_data["labels"]
            for pred_data in all_predictions.values()
        ],
        labels=list(all_predictions.keys()),
    )
    ax4.set_title("Accuracy Distribution", fontweight="bold")
    ax4.set_ylabel("Correct Predictions")
    ax4.tick_params(axis="x", rotation=45)

    # 5. Confusion Matrix Comparison (for first 3 models)
    models_to_show = min(3, len(all_predictions))
    for i, (model_name, pred_data) in enumerate(
        list(all_predictions.items())[:models_to_show]
    ):
        ax = fig.add_subplot(gs[1, 2 + i] if i < 2 else gs[2, i - 2])

        cm = confusion_matrix(pred_data["labels"], pred_data["predictions"])
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # avoid divide-by-zero for classes absent from test set
        cm_normalized = cm.astype("float") / row_sums

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names[:5] if len(class_names) > 5 else class_names,
            yticklabels=class_names[:5] if len(class_names) > 5 else class_names,
            cbar=False,
        )
        ax.set_title(f"{model_name}\nNormalized CM", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # 6. Model Rankings Table
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis("off")

    # Create ranking table
    rankings = df.copy()
    rankings["Rank_Accuracy"] = rankings["Accuracy"].rank(ascending=False, method="min")
    rankings["Rank_F1"] = rankings["F1 Score"].rank(ascending=False, method="min")
    rankings["Rank_Loss"] = rankings["Final Loss"].rank(ascending=True, method="min")
    rankings["Average_Rank"] = (
        rankings["Rank_Accuracy"] + rankings["Rank_F1"] + rankings["Rank_Loss"]
    ) / 3
    rankings = rankings.sort_values("Average_Rank")

    # Create table text
    table_text = "🏆 Model Rankings\n" + "=" * 30 + "\n\n"
    for i, (_, row) in enumerate(rankings.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        table_text += f"{medal} {row['Model']}\n"
        table_text += f"   Acc: {row['Accuracy']:.4f} (#{int(row['Rank_Accuracy'])})\n"
        table_text += f"   F1:  {row['F1 Score']:.4f} (#{int(row['Rank_F1'])})\n"
        table_text += f"   Loss: {row['Final Loss']:.4f} (#{int(row['Rank_Loss'])})\n\n"

    ax6.text(
        0.1,
        0.9,
        table_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
    )

    # Save comparison dashboard
    dashboard_path = os.path.join(comp_dir, "model_comparison_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    print(f"* Model comparison dashboard saved to {dashboard_path}")

    plt.tight_layout()
    plt.show()
    plt.close()

    # Save detailed comparison CSV
    csv_path = os.path.join(comp_dir, "model_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Model comparison results saved to {csv_path}")

    # Save ranking results
    ranking_path = os.path.join(comp_dir, "model_rankings.csv")
    rankings.to_csv(ranking_path, index=False)
    print(f"🏆 Model rankings saved to {ranking_path}")


def generate_evaluation_report(model_name, metrics, cm, class_names, results_dir):
    """
    Generate a comprehensive evaluation report in multiple formats.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    # Create report directory
    report_dir = os.path.join(results_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .section {{ margin: 30px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Evaluation Report</h1>
            <h2>{model_name}</h2>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section">
            <h3>Overall Performance</h3>
    """

    for metric, value in metrics["overall"].items():
        html_content += (
            f'<div class="metric"><strong>{metric}:</strong> {value:.4f}</div>\n'
        )

    html_content += """
        </div>
        
        <div class="section">
            <h3>Per-Class Performance</h3>
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>
    """

    for class_name, class_metrics in metrics["per_class"].items():
        html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{class_metrics['Precision']:.4f}</td>
                    <td>{class_metrics['Recall']:.4f}</td>
                    <td>{class_metrics['F1 Score']:.4f}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h3>Confusion Matrix</h3>
            <p>Please refer to the generated confusion matrix images for detailed analysis.</p>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    html_path = os.path.join(
        report_dir, f"{model_name}_evaluation_report_{timestamp}.html"
    )
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"📄 HTML evaluation report saved to {html_path}")

    # Generate detailed text report
    text_report = f"""
MODEL EVALUATION REPORT
{model_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*60}

OVERALL PERFORMANCE:
{'-'*20}
"""

    for metric, value in metrics["overall"].items():
        text_report += f"{metric:<15}: {value:.6f}\n"

    text_report += f"\nPER-CLASS PERFORMANCE:\n{'-'*20}\n"

    for class_name, class_metrics in metrics["per_class"].items():
        text_report += f"\n{class_name}:\n"
        for metric, value in class_metrics.items():
            text_report += f"  {metric:<12}: {value:.6f}\n"

    text_report += f"\nCONFUSION MATRIX:\n{'-'*20}\n"
    text_report += "Classes: " + ", ".join(class_names) + "\n\n"
    text_report += str(cm) + "\n"

    # Calculate additional statistics
    total_samples = np.sum(cm)
    correct_predictions = np.trace(cm)
    text_report += f"\nSTATISTICS:\n{'-'*20}\n"
    text_report += f"Total Samples: {total_samples}\n"
    text_report += f"Correct Predictions: {correct_predictions}\n"
    text_report += f"Incorrect Predictions: {total_samples - correct_predictions}\n"
    text_report += f"Overall Accuracy: {correct_predictions / total_samples:.6f}\n"

    # Save text report
    text_path = os.path.join(
        report_dir, f"{model_name}_evaluation_report_{timestamp}.txt"
    )
    with open(text_path, "w") as f:
        f.write(text_report)
    print(f"📄 Text evaluation report saved to {text_path}")

    return html_path, text_path


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    patience=5,
    checkpoint_dir="checkpoints",
):
    start_time = time.time()

    # Lists to store loss and accuracy for each epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    model.to(device)

    # Initialize EarlyStopping object
    early_stopping = EarlyStopping(
        patience=patience, delta=0, save_best_model=True, checkpoint_dir=checkpoint_dir
    )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy and loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # Calculate average loss and accuracy for training
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        # Calculate average loss and accuracy for validation
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Call early stopping
        early_stopping(val_loss, model, epoch)

        # If early stopping triggered, break the loop
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load the best model after training
    early_stopping.load_best_model(model)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s")

    return history


def plot_training_history(history, save_path=None, model_name="Model"):
    """
    Create comprehensive training history plots with enhanced styling.

    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the plots
        model_name: Name of the model for the plot title
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{model_name} Training History", fontsize=16, fontweight="bold")

    # Plot 1: Training and validation loss
    ax1.plot(
        epochs,
        history["train_loss"],
        "b-",
        label="Training Loss",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        "r-",
        label="Validation Loss",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    # Highlight best validation loss
    best_val_loss_idx = history["val_loss"].index(min(history["val_loss"]))
    ax1.plot(
        epochs[best_val_loss_idx],
        history["val_loss"][best_val_loss_idx],
        "ro",
        markersize=8,
        label=f"Best Val Loss (Epoch {epochs[best_val_loss_idx]})",
    )

    ax1.set_xlabel("Epochs", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training and validation accuracy
    train_acc_pct = [acc * 100 for acc in history["train_accuracy"]]
    val_acc_pct = [acc * 100 for acc in history["val_accuracy"]]

    ax2.plot(
        epochs,
        train_acc_pct,
        "b-",
        label="Training Accuracy",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax2.plot(
        epochs,
        val_acc_pct,
        "r-",
        label="Validation Accuracy",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    # Highlight best validation accuracy
    best_val_acc_idx = history["val_accuracy"].index(max(history["val_accuracy"]))
    ax2.plot(
        epochs[best_val_acc_idx],
        val_acc_pct[best_val_acc_idx],
        "ro",
        markersize=8,
        label=f"Best Val Acc (Epoch {epochs[best_val_acc_idx]})",
    )

    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Epochs", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss difference (overfitting indicator)
    loss_diff = [
        val - train for val, train in zip(history["val_loss"], history["train_loss"])
    ]
    ax3.plot(epochs, loss_diff, "g-", linewidth=2, marker="d", markersize=3)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Epochs", fontweight="bold")
    ax3.set_ylabel("Loss Difference", fontweight="bold")
    ax3.set_title(
        "Validation - Training Loss\n(Overfitting Indicator)", fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)

    # Add annotation
    ax3.text(
        0.02,
        0.98,
        "Positive = Overfitting\nNegative = Underfitting",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Plot 4: Training summary statistics
    ax4.axis("off")

    # Calculate summary statistics
    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    final_train_acc = history["train_accuracy"][-1] * 100
    final_val_acc = history["val_accuracy"][-1] * 100
    best_val_acc = max(history["val_accuracy"]) * 100
    best_val_loss = min(history["val_loss"])

    # Create summary text
    summary_text = f"""
Training Summary:

* Final Metrics:
   - Training Loss: {final_train_loss:.4f}
   - Validation Loss: {final_val_loss:.4f}
   - Training Accuracy: {final_train_acc:.2f}%
   - Validation Accuracy: {final_val_acc:.2f}%

* Best Performance:
   - Best Validation Accuracy: {best_val_acc:.2f}%
   - Best Validation Loss: {best_val_loss:.4f}

* Training Progress:
   - Total Epochs: {len(epochs)}
   - Loss Improvement: {(history["train_loss"][0] - final_train_loss):.4f}
   - Accuracy Improvement: {(final_train_acc - history["train_accuracy"][0]*100):.2f}%
"""

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def create_training_animation(history, save_path=None, model_name="Model"):
    """
    Create an animated plot showing training progress over time.

    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the animation (as GIF)
        model_name: Name of the model for the plot title
    """
    try:
        from matplotlib.animation import FuncAnimation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{model_name} Training Progress Animation", fontsize=14, fontweight="bold"
        )

        def animate(frame):
            ax1.clear()
            ax2.clear()

            epochs = range(1, frame + 2)

            # Plot loss up to current frame
            ax1.plot(
                epochs,
                history["train_loss"][: frame + 1],
                "b-",
                label="Training Loss",
                linewidth=2,
            )
            ax1.plot(
                epochs,
                history["val_loss"][: frame + 1],
                "r-",
                label="Validation Loss",
                linewidth=2,
            )
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"Loss Progress (Epoch {frame+1})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot accuracy up to current frame
            train_acc = [acc * 100 for acc in history["train_accuracy"][: frame + 1]]
            val_acc = [acc * 100 for acc in history["val_accuracy"][: frame + 1]]
            ax2.plot(epochs, train_acc, "b-", label="Training Accuracy", linewidth=2)
            ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy", linewidth=2)
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title(f"Accuracy Progress (Epoch {frame+1})")
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(history["train_loss"]),
            interval=500,
            repeat=True,
            blit=False,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=2)
            print(f"Training animation saved to {save_path}")

        plt.show()
        return anim

    except ImportError:
        print("Animation requires matplotlib.animation. Skipping animation creation.")
        return None


def evaluate_model(
    model, criterion, test_loader, class_names, device, results_dir="./results"
):
    """
    Enhanced model evaluation with comprehensive visualization and metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function.
        test_loader (DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names for the confusion matrix.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').
        results_dir (str): Directory to save the evaluation results.

    Returns:
        tuple: Confusion matrix, performance metrics table, and final loss.
    """
    import numpy as np
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    start_time = time.time()
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    # Create evaluation results directory (per-model to avoid overwrites)
    model_name = _get_model_display_name(model)
    eval_dir = os.path.join(results_dir, "evaluation", model_name)
    os.makedirs(eval_dir, exist_ok=True)

    print("🔬 Running model evaluation...")

    # Evaluation with progress bar
    eval_pbar = tqdm(test_loader, desc="Evaluating", ncols=100)

    # Variables for efficiency metrics
    inference_times = []

    # Evaluate the model
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(eval_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Measure inference time for this batch
            batch_start = time.time()
            outputs = model(inputs)
            batch_end = time.time()
            inference_times.append(batch_end - batch_start)

            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item() * inputs.size(0)

            # Update progress bar
            current_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(
                all_preds
            )
            eval_pbar.set_postfix(
                {"Acc": f"{current_acc:.4f}", "Loss": f"{loss.item():.4f}"}
            )

    # Calculate average loss
    final_loss = total_loss / len(test_loader.dataset)

    # Calculate Efficiency Metrics
    avg_batch_time = np.mean(inference_times)
    latency = avg_batch_time  # in seconds per batch
    throughput = test_loader.batch_size / avg_batch_time  # in samples per second

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print(f"\n* Evaluation Results:")
    print(f"   • Final Loss: {final_loss:.4f}")
    print(f"   • Overall Accuracy: {(all_preds == all_labels).mean():.4f}")
    print(f"   • Latency: {latency*1000:.2f} ms/batch")
    print(f"   • Throughput: {throughput:.2f} samples/sec")

    # Calculate comprehensive metrics
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    metrics = calculate_comprehensive_metrics(
        all_labels, all_preds, all_probs, class_names
    )
    
    # Add efficiency metrics to the metrics dictionary
    metrics["efficiency"] = {
        "Latency (ms/batch)": latency * 1000,
        "Throughput (samples/sec)": throughput,
        "Avg Inference Time (ms/batch)": avg_batch_time * 1000
    }

    # Create comprehensive evaluation dashboard
    create_evaluation_dashboard(
        all_labels,
        all_preds,
        all_probs,
        class_names,
        model_name,
        eval_dir,
        metrics,
        final_loss,
    )

    # Save detailed classification report. Restrict to the evaluated classes
    # (those present as true labels) so that the report's "macro avg" row uses
    # the same convention as the manuscript (unweighted mean over the K
    # evaluated classes) instead of including zero-support classes.
    evaluated = np.unique(np.asarray(all_labels))
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[class_names[i] for i in evaluated],
        labels=evaluated,
        output_dict=True,
        zero_division=0,
    )
    save_classification_report(report, model_name, eval_dir)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"⏱️  Evaluation completed in: {total_time // 60:.0f}m {total_time % 60:.0f}s")

    return cm, metrics, final_loss


def calculate_comprehensive_metrics(labels, preds, probs, class_names):
    """
    Calculate comprehensive evaluation metrics including per-class metrics.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    # Overall metrics. Precision, recall and F1 are macro-averaged over the
    # evaluated classes, i.e. the classes that occur as true labels. Classes
    # absent from the evaluation (zero support) are excluded, so the unweighted
    # mean matches the manuscript convention (macro average over the K
    # evaluated classes). A support-weighted average would instead be dominated
    # by the large classes.
    accuracy = accuracy_score(labels, preds)
    evaluated = np.unique(np.asarray(labels))

    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None, labels=range(len(class_names)), zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, labels=range(len(class_names)), zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, labels=range(len(class_names)), zero_division=0)

    precision = float(np.mean(precision_per_class[evaluated]))
    recall = float(np.mean(recall_per_class[evaluated]))
    f1 = float(np.mean(f1_per_class[evaluated]))

    # Multi-class ROC AUC
    try:
        if len(np.unique(labels)) > 2:
            # Multi-class case
            labels_binarized = label_binarize(labels, classes=range(len(class_names)))
            roc_auc = roc_auc_score(
                labels_binarized, probs, average="weighted", multi_class="ovr"
            )
        else:
            # Binary case
            roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        roc_auc = 0.0

    return {
        "overall": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
        },
        "per_class": {
            class_names[i]: {
                "Precision": precision_per_class[i],
                "Recall": recall_per_class[i],
                "F1 Score": f1_per_class[i],
            }
            for i in range(len(class_names))
        },
    }


def create_evaluation_dashboard(
    labels, preds, probs, class_names, model_name, save_dir, metrics, final_loss
):
    """
    Create a comprehensive evaluation dashboard with multiple visualizations.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle(
        f"{model_name} - Comprehensive Evaluation Dashboard",
        fontsize=20,
        fontweight="bold",
    )

    # 1. Confusion Matrix (normalized)
    ax1 = fig.add_subplot(gs[0, 0:2])
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_title("Normalized Confusion Matrix", fontweight="bold")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # 2. Confusion Matrix (absolute numbers)
    ax2 = fig.add_subplot(gs[0, 2:4])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
    )
    ax2.set_title("Confusion Matrix (Counts)", fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    # 3. Per-class Performance Bar Chart
    ax3 = fig.add_subplot(gs[1, 0:2])
    class_metrics = metrics["per_class"]
    classes = list(class_metrics.keys())
    precision_vals = [class_metrics[cls]["Precision"] for cls in classes]
    recall_vals = [class_metrics[cls]["Recall"] for cls in classes]
    f1_vals = [class_metrics[cls]["F1 Score"] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    ax3.bar(x - width, precision_vals, width, label="Precision", alpha=0.8)
    ax3.bar(x, recall_vals, width, label="Recall", alpha=0.8)
    ax3.bar(x + width, f1_vals, width, label="F1 Score", alpha=0.8)

    ax3.set_xlabel("Classes")
    ax3.set_ylabel("Score")
    ax3.set_title("Per-Class Performance Metrics", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Class Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))

    wedges, texts, autotexts = ax4.pie(
        counts,
        labels=[class_names[i] for i in unique],
        autopct="%1.1f%%",
        colors=colors,
    )
    ax4.set_title("True Class Distribution", fontweight="bold")

    # 5. Prediction Distribution
    ax5 = fig.add_subplot(gs[1, 3])
    unique_pred, counts_pred = np.unique(preds, return_counts=True)

    wedges, texts, autotexts = ax5.pie(
        counts_pred,
        labels=[class_names[i] for i in unique_pred],
        autopct="%1.1f%%",
        colors=colors,
    )
    ax5.set_title("Predicted Class Distribution", fontweight="bold")

    # 6. Overall Metrics Summary
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.axis("off")

    overall_metrics = metrics["overall"]
    summary_text = f"""
Overall Performance Summary:

* Accuracy: {overall_metrics['Accuracy']:.4f}
* Precision: {overall_metrics['Precision']:.4f}
* Recall: {overall_metrics['Recall']:.4f}
* F1 Score: {overall_metrics['F1 Score']:.4f}
* ROC AUC: {overall_metrics['ROC AUC']:.4f}
* Final Loss: {final_loss:.4f}

Dataset Info:
   - Total Samples: {len(labels)}
   - Number of Classes: {len(class_names)}
   - Correct Predictions: {sum(labels == preds)}
   - Wrong Predictions: {sum(labels != preds)}
"""

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )

    # 7. ROC Curves (for binary or simplified multi-class)
    ax7 = fig.add_subplot(gs[2, 1:3])
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            ax7.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
        else:
            # Multi-class - show average
            labels_binarized = label_binarize(labels, classes=range(len(class_names)))
            for i in range(
                min(5, len(class_names))
            ):  # Show max 5 classes to avoid clutter
                fpr, tpr, _ = roc_curve(labels_binarized[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax7.plot(
                    fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.2f})"
                )

        ax7.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.5)
        ax7.set_xlim([0.0, 1.0])
        ax7.set_ylim([0.0, 1.05])
        ax7.set_xlabel("False Positive Rate")
        ax7.set_ylabel("True Positive Rate")
        ax7.set_title("ROC Curves", fontweight="bold")
        ax7.legend(loc="lower right")
        ax7.grid(True, alpha=0.3)

    except Exception as e:
        ax7.text(
            0.5,
            0.5,
            f"ROC curve not available\n{str(e)}",
            ha="center",
            va="center",
            transform=ax7.transAxes,
        )
        ax7.set_title("ROC Curves (Not Available)", fontweight="bold")

    # 8. Prediction Confidence Distribution
    ax8 = fig.add_subplot(gs[2, 3])
    max_probs = np.max(probs, axis=1)
    correct_preds = labels == preds

    ax8.hist(
        max_probs[correct_preds], bins=20, alpha=0.7, label="Correct", color="green"
    )
    ax8.hist(
        max_probs[~correct_preds], bins=20, alpha=0.7, label="Incorrect", color="red"
    )
    ax8.set_xlabel("Prediction Confidence")
    ax8.set_ylabel("Count")
    ax8.set_title("Prediction Confidence Distribution", fontweight="bold")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Save the dashboard
    dashboard_path = os.path.join(save_dir, f"{model_name}_evaluation_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    print(f"* Evaluation dashboard saved to {dashboard_path}")

    # Also show the plot - handle tight_layout warnings gracefully
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.tight_layout()
        except:
            pass  # If tight_layout fails, continue without it
    plt.show()
    plt.close()


def save_classification_report(report_dict, model_name, save_dir):
    """
    Save detailed classification report as CSV and visualization.
    """
    import pandas as pd

    # Convert report to DataFrame
    df = pd.DataFrame(report_dict).transpose()

    # Save as CSV
    csv_path = os.path.join(save_dir, f"{model_name}_classification_report.csv")
    df.to_csv(csv_path)
    print(f"📄 Classification report saved to {csv_path}")

    # Create visualization of the report
    fig, ax = plt.subplots(figsize=(10, 8))

    # Remove summary rows for visualization
    df_viz = df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

    # Create heatmap
    sns.heatmap(
        df_viz[["precision", "recall", "f1-score"]],
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        ax=ax,
    )
    ax.set_title(f"{model_name} - Classification Report Heatmap", fontweight="bold")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Classes")

    # Save visualization
    viz_path = os.path.join(save_dir, f"{model_name}_classification_report_heatmap.png")
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    print(f"* Classification report heatmap saved to {viz_path}")
    plt.close()


def save_confusion_matrix(cm, class_names, model_name, results_dir):
    """
    Enhanced confusion matrix visualization with multiple views and better styling.

    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.
        model_name (str): Name of the model.
        results_dir (str): Directory to save the plot.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Ensure the results directory exists
    confusion_dir = os.path.join(results_dir, "confusion_matrix")
    os.makedirs(confusion_dir, exist_ok=True)

    # Set style for better plots
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with multiple confusion matrix views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{model_name} - Confusion Matrix Analysis", fontsize=16, fontweight="bold"
    )

    # 1. Raw counts confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_xlabel("Predicted Label", fontweight="bold")
    ax1.set_ylabel("True Label", fontweight="bold")
    ax1.set_title("Raw Counts", fontweight="bold")

    # 2. Normalized confusion matrix (by true class)
    cm_normalized = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={"label": "Proportion"},
    )
    ax2.set_xlabel("Predicted Label", fontweight="bold")
    ax2.set_ylabel("True Label", fontweight="bold")
    ax2.set_title("Normalized by True Class", fontweight="bold")

    # 3. Precision-focused view (normalized by predicted class)
    cm_precision = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    sns.heatmap(
        cm_precision,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax3,
        cbar_kws={"label": "Precision"},
    )
    ax3.set_xlabel("Predicted Label", fontweight="bold")
    ax3.set_ylabel("True Label", fontweight="bold")
    ax3.set_title("Precision View (Normalized by Predicted)", fontweight="bold")

    # 4. Error analysis heatmap
    # Calculate misclassification rate for each class pair
    total_samples = cm.sum()
    error_matrix = cm.copy().astype("float")
    np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
    error_rate = error_matrix / total_samples

    sns.heatmap(
        error_rate,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax4,
        cbar_kws={"label": "Error Rate"},
    )
    ax4.set_xlabel("Predicted Label", fontweight="bold")
    ax4.set_ylabel("True Label", fontweight="bold")
    ax4.set_title("Error Analysis (Misclassification Rates)", fontweight="bold")

    plt.tight_layout()

    # Save the comprehensive confusion matrix plot
    plot_path = os.path.join(
        confusion_dir, f"{model_name}_confusion_matrix_comprehensive.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"* Comprehensive confusion matrix saved to {plot_path}")
    plt.show()
    plt.close()

    # Create additional individual high-quality plots

    # High-quality normalized confusion matrix
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(cm_normalized)

    # Create heatmap with better styling
    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Recall (True Positive Rate)"},
    )

    # Add title and labels
    plt.title(
        f"{model_name} - Normalized Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Add accuracy information
    overall_accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(
        0.02,
        0.02,
        f"Overall Accuracy: {overall_accuracy:.3f}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    # Save high-quality plot
    hq_plot_path = os.path.join(
        confusion_dir, f"{model_name}_confusion_matrix_normalized_hq.png"
    )
    plt.savefig(hq_plot_path, dpi=300, bbox_inches="tight")
    print(f"* High-quality normalized confusion matrix saved to {hq_plot_path}")
    plt.close()

    # Create per-class binary confusion matrices with improved layout
    n_classes = len(class_names)
    cols = min(4, n_classes)
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    fig.suptitle(
        f"{model_name} - Per-Class Binary Confusion Matrices",
        fontsize=16,
        fontweight="bold",
    )

    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, class_name in enumerate(class_names):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Create binary confusion matrix for this class
        binary_cm = np.zeros((2, 2), dtype=int)

        # True positives
        binary_cm[0, 0] = cm[i, i]
        # False negatives
        binary_cm[0, 1] = np.sum(cm[i, :]) - cm[i, i]
        # False positives
        binary_cm[1, 0] = np.sum(cm[:, i]) - cm[i, i]
        # True negatives
        binary_cm[1, 1] = (
            np.sum(cm) - binary_cm[0, 0] - binary_cm[0, 1] - binary_cm[1, 0]
        )

        # Calculate class-specific metrics
        precision = (
            binary_cm[0, 0] / (binary_cm[0, 0] + binary_cm[1, 0])
            if (binary_cm[0, 0] + binary_cm[1, 0]) > 0
            else 0
        )
        recall = (
            binary_cm[0, 0] / (binary_cm[0, 0] + binary_cm[0, 1])
            if (binary_cm[0, 0] + binary_cm[0, 1]) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Create heatmap
        sns.heatmap(
            binary_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=[f"{class_name}", "Others"],
            yticklabels=[f"{class_name}", "Others"],
            cbar=False,
            square=True,
        )

        ax.set_title(
            f"{class_name}\nP:{precision:.2f} R:{recall:.2f} F1:{f1:.2f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # Hide empty subplots
    for i in range(n_classes, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()

    # Save per-class plot
    per_class_path = os.path.join(
        confusion_dir, f"{model_name}_confusion_matrix_per_class.png"
    )
    plt.savefig(per_class_path, dpi=300, bbox_inches="tight")
    print(f"* Per-class confusion matrices saved to {per_class_path}")
    plt.close()

    # Generate confusion matrix statistics
    stats_path = os.path.join(confusion_dir, f"{model_name}_confusion_matrix_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Confusion Matrix Statistics for {model_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Raw Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

        f.write("Normalized Confusion Matrix (by true class):\n")
        f.write(str(cm_normalized) + "\n\n")

        f.write("Class-wise Statistics:\n")
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            f.write(f"\n{class_name}:\n")
            f.write(f"  True Positives: {tp}\n")
            f.write(f"  False Negatives: {fn}\n")
            f.write(f"  False Positives: {fp}\n")
            f.write(f"  True Negatives: {tn}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall (Sensitivity): {recall:.4f}\n")
            f.write(f"  Specificity: {specificity:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")

        overall_accuracy = np.trace(cm) / np.sum(cm)
        f.write(f"\nOverall Accuracy: {overall_accuracy:.4f}\n")

    print(f"📄 Confusion matrix statistics saved to {stats_path}")


def print_metrics(metrics):
    """
    Enhanced metrics printing with better formatting.
    """
    print("\n* Model Performance Metrics:")
    print("=" * 40)

    if "overall" in metrics:
        print("\n* Overall Performance:")
        for metric, value in metrics["overall"].items():
            print(f"   • {metric:<12}: {value:.4f}")
    else:
        # Legacy format support
        for metric, value in metrics.items():
            print(f"   • {metric:<12}: {value:.4f}")

    if "per_class" in metrics:
        print("\n* Per-Class Performance:")
        for class_name, class_metrics in metrics["per_class"].items():
            print(f"\n   {class_name}:")
            for metric, value in class_metrics.items():
                print(f"     • {metric:<10}: {value:.4f}")

    print("=" * 40)


def save_model(model, optimizer, num_epochs, final_loss):
    # Save a full checkpoint with additional details
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # Optional if you need it
        "epoch": num_epochs,
        "loss": final_loss,  # Replace final_loss with your final loss value
    }
    torch.save(
        checkpoint,
        os.path.join(CHECKPOINT_DIR, f"{model.__class__.__name__}-{dt_string.replace('/', '-').replace(':', '-')}.pth"),
    )


def calculate_inference_time(model, test_loader, device="cpu"):
    """
    Calculate the inference time of a PyTorch model.

    Parameters:
    - model: torch.nn.Module, the PyTorch model to evaluate
    - test_loader: DataLoader, the test dataset loader
    - device: str, device to run the inference ('cuda' or 'cpu')

    Returns:
    - float, average inference time in milliseconds
    """
    model = model.to(device)
    model.eval()

    # Warm-up runs to stabilize CUDA performance
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # Run warm-up on just one batch

    # Measure inference time across the dataset with a progress bar
    times = []
    with torch.no_grad():
        for inputs, _ in tqdm(
            test_loader, desc="Calculating Inference Time", unit="batch"
        ):
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Return the average inference time per batch
    return sum(times) / len(times)


def export_confusion_matrix_to_csv(
    cm, class_names, save_path="./csv", filename="confusion_matrix.csv"
):
    """
    Export a confusion matrix to a CSV file with proper row and column labels.

    Parameters:
    -----------
    cm : numpy.ndarray
        The confusion matrix to export
    class_names : list
        List of class names for labeling rows and columns
    save_path : str, optional
        Directory path where the CSV file will be saved
    filename : str, optional
        Name of the CSV file

    Returns:
    --------
    str
        Path to the saved CSV file
    """
    import os
    import numpy as np
    import pandas as pd

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create a DataFrame with proper labeling
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Add row labels for clarity (actual classes)
    cm_df.index.name = "Actual"

    # Add a custom header for the first column (predicted classes)
    full_path = os.path.join(save_path, filename)

    # Save to CSV
    cm_df.to_csv(full_path)
    print(f"Confusion matrix saved to {full_path}")

    return full_path
