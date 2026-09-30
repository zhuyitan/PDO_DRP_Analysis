"""
Model-specific params (Model: LightGBM)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = [
    {"name": "use_lincs",
     "type": str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression and Mordred descriptors data.",
    },
    {
        "name": "ge_scaler_fname",
        "type": str,
        "default": "x_data_gene_expression_scaler.gz",
        "help": "File name to save the gene expression scaler object.",
    },

    {
        "name": "md_scaler_fname",
        "type": str,
        "default": "x_data_mordred_scaler.gz",
        "help": "File name to save the Mordred scaler object.",
    },

    {
        "name": "pretrained_ge_scaler",
        "type": str,
        "default": None,
        "help": (
            "Path to a saved gene-expression scaler from pretraining. "
            "If supplied, the scaler is loaded instead of fitting a new one."
        ),
    },

    {
        "name": "pretrained_md_scaler",
        "type": str,
        "default": None,
        "help": (
            "Path to a saved Mordred scaler from pretraining. "
            "If supplied, the scaler is loaded instead of fitting a new one."
        ),
    },
]


train_params = [
    {"name": "n_estimators",
     "type": int,
     "default": 10000,
     "help": "Number of estimators."
    },
    {"name": "max_depth",
     "type": int,
     "default": 5,
     "help": "Max depth."
    },
    {"name": "num_leaves",
     "type": int,
     "default": 16,
     "help": "Number of leaves."
    },
    {"name": "min_data_in_leaf",
     "type": int,
     "default": 128
    },
    {"name": "pretrained_model",
     "type": str,
     "default": None
    },
    {"name": "transfer_rounds",
     "type": int,
     "default": 1000
    }
]


infer_params = []