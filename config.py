"""
BLoco Model Configuration
"""
import os

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")
BUG_REPORTS_DIR = os.path.join(DATASET_DIR, "bug reports")
SOURCE_FILES_DIR = os.path.join(DATASET_DIR, "source files")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# Project Configurations
# ============================================================
PROJECTS = {
    "AspectJ": {
        "bug_report_xml": os.path.join(BUG_REPORTS_DIR, "AspectJ.xml"),
        "source_dir": os.path.join(SOURCE_FILES_DIR, "org.aspectj-bug433351"),
        "db_name": "aspectj",
    },
    "Birt": {
        "bug_report_xml": os.path.join(BUG_REPORTS_DIR, "Birt.xml"),
        "source_dir": os.path.join(SOURCE_FILES_DIR, "birt-20140211-1400"),
        "db_name": "birt",
    },
    "Eclipse_Platform_UI": {
        "bug_report_xml": os.path.join(BUG_REPORTS_DIR, "Eclipse_Platform_UI.xml"),
        "source_dir": os.path.join(SOURCE_FILES_DIR, "eclipse.platform.ui-johna-402445"),
        "db_name": "eclipse_platform_ui",
    },
    "SWT": {
        "bug_report_xml": os.path.join(BUG_REPORTS_DIR, "SWT.xml"),
        "source_dir": os.path.join(SOURCE_FILES_DIR, "eclipse.platform.swt-xulrunner-31"),
        "db_name": "swt",
    },
    "Tomcat": {
        "bug_report_xml": os.path.join(BUG_REPORTS_DIR, "Tomcat.xml"),
        "source_dir": os.path.join(SOURCE_FILES_DIR, "tomcat-7.0.51"),
        "db_name": "tomcat",
    },
}

# ============================================================
# Model Hyperparameters
# ============================================================
# Embedding
VOCAB_SIZE = 50000
EMBED_DIM = 200
MAX_SEQ_LEN = 512          # max tokens per clue / code file

# TextCNN (Bug Report Encoder)
TEXTCNN_KERNEL_SIZES = [2, 3, 4, 5]
TEXTCNN_NUM_FILTERS = 100  # per kernel size

# Code Encoder (Option A: TextCNN/GRU)
CODE_HIDDEN_DIM = 200
CODE_GRU_LAYERS = 2

# Bi-Affine
BIAFFINE_DIM = EMBED_DIM   # same as embed_dim

# FFN
FFN_HIDDEN_DIM = 256

# ============================================================
# Training
# ============================================================
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOP_PATIENCE = 5
NEGATIVE_SAMPLE_RATIO = 10     # number of negative files per positive
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================
# Evaluation Metrics
# ============================================================
TOP_K_VALUES = [1, 5, 10]

# ============================================================
# Device
# ============================================================
DEVICE = "cuda"  # will fallback to cpu in code if not available
