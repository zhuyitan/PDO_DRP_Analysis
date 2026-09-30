"""
Model-specific params (Model: GraphDRP)

If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = [
    {
        "name": "use_lincs",
        "type": str2bool,
        "default": True,
        "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {
        "name": "scaling",
        "type": str,
        "default": "std",
        "choices": ["std", "minmax", "miabs", "robust"],
        "help": "Scaler for gene expression data.",
    },
    {
        "name": "ge_scaler_fname",
        "type": str,
        "default": "x_data_gene_expression_scaler.gz",
        "help": "File name to save the gene expression scaler object.",
    },
    {
        "name": "pretrained_ge_scaler",
        "type": str,
        "default": None,
        "help": (
            "Optional path to a pretrained gene-expression scaler. "
            "When provided, preprocessing loads this scaler instead of "
            "fitting a new scaler on the current data."
        ),
    },
]

train_params = [
    {
        "name": "model_arch",
        "type": str,
        "default": "GINConvNet",
        "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
        "help": "Model architecture to run.",
    },
    {
        "name": "log_interval",
        "type": int,
        "default": 20,
        "help": "Interval for saving o/p",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Cuda device (e.g.: cuda:0, cuda:1).",
    },
    {
        "name": "pretrained_model",
        "type": str,
        "default": "None",
    },
    {
        "name": "freeze_layers",
        "type": str,
        "default": "None",
        "choices": [
            "None",
            "Freeze Drug",
            "Freeze Cell",
            "Freeze Both",
        ],
        "help": (
            "Transfer-learning encoder freezing mode. 'None' trains both "
            "encoders; 'Freeze Drug' freezes the GIN drug encoder; "
            "'Freeze Cell' freezes the cancer-cell encoder; and "
            "'Freeze Both' freezes both encoders."
        ),
    },
    {
        "name": "freeze_drug_layers",
        "type": int,
        "default": -1,
        "help": (
            "Number of drug-encoder stages to freeze from the input side. "
            "Valid values are 0-6; -1 freezes all 6 stages."
        ),
    },
    {
        "name": "freeze_cell_layers",
        "type": int,
        "default": -1,
        "help": (
            "Number of cancer-cell encoder stages to freeze from the input side. "
            "Valid values are 0-4; -1 freezes all 4 stages."
        ),
    },
    {
        "name": "freeze_head_layers",
        "type": str,
        "default": "None",
        "choices": [
            "None",
            "Freeze FC1",
            "Freeze FC1 FC2",
            "Freeze Head",
        ],
        "help": (
            "Prediction-head freezing mode. 'None' trains fc1, fc2, and out; "
            "'Freeze FC1' freezes the 256->1024 layer; "
            "'Freeze FC1 FC2' freezes both hidden linear layers and trains "
            "only the 128->1 output layer; 'Freeze Head' freezes the entire "
            "prediction head."
        ),
    },
    {
        "name": "dropout",
        "type": float,
        "default": 0.5,
        "help": (
            "Dropout probability applied after the hidden layers "
            "of the GraphDRP prediction head."
        ),
    }
      
]


infer_params = [
    {
        "name": "model_arch",
        "type": str,
        "default": "GINConvNet",
        "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
        "help": "Model architecture to run.",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Cuda device (e.g.: cuda:0, cuda:1).",
    },
]
