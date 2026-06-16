"""
Configuration for Bosnian AI Text Detector using BERTić
Model: CLASSLA/bcms-bertic (Electra-based, trained on Bosnian/Croatian/Montenegrin/Serbian)
Task: Binary classification — human-written (0) vs AI-generated/rewritten (1)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Path to your collected news articles (CSV with columns: text, label)
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Column names in your CSV files
    text_column: str = "text"
    label_column: str = "label"          # 0 = human-written, 1 = AI-generated

    # Dataset split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Random seed for reproducibility
    random_seed: int = 42

    # Minimum / maximum article length in characters (filter too-short scraped junk)
    min_text_length: int = 200
    max_text_length: int = 50_000


@dataclass
class ModelConfig:
    # BERTić — Electra discriminator trained on BCMS (Bosnian/Croatian/Montenegrin/Serbian)
    # This is the key model from the BERTić paper (Ljubešić & Lauc, 2021)
    model_name: str = "CLASSLA/bcms-bertic"

    # Tokenizer settings
    max_length: int = 512        # BERT/Electra architectural limit
    do_lower_case: bool = False  # BERTić is cased — Bosnian diacritics (č, ć, š, ž, đ) matter

    # Classification
    num_labels: int = 2
    problem_type: str = "single_label_classification"


@dataclass
class TrainingConfig:
    output_dir: str = "output/bertic_bosnian_detector"
    logging_dir: str = "output/logs"

    # Training hyperparameters (following Turkish BERT paper approach)
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2   # Effective batch size = 8 * 2 = 16

    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_ratio: float = 0.1             # 10% of total steps for warmup

    # Regularization
    hidden_dropout_prob: float = 0.2
    attention_probs_dropout_prob: float = 0.2
    label_smoothing_factor: float = 0.1

    # Early stopping
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 2

    # Evaluation strategy
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # Mixed precision (speeds up training, use False if no GPU)
    fp16: bool = True

    # Logging
    logging_steps: int = 50
    report_to: str = "none"    # Change to "wandb" if you want experiment tracking

    # Reproducibility
    seed: int = 42
    num_train_runs: int = 5    # Run 5 times, report average (as in BERTić paper)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Create output directories
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_data_dir).mkdir(parents=True, exist_ok=True)


# Singleton config instance
cfg = Config()
