import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


class GINConvNet(torch.nn.Module):
    """GIN-based GraphDRP model with optional encoder freezing."""

    FREEZE_MODES = (
        "None",
        "Freeze Drug",
        "Freeze Cell",
        "Freeze Both",
    )

    HEAD_FREEZE_MODES = (
        "None",
        "Freeze FC1",
        "Freeze FC1 FC2",
        "Freeze Head",
    )

    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        num_genes=942,
        in_dim=None,
    ):
        """
        If loading a previously saved model, ``in_dim`` should be provided.
        This avoids a dummy forward pass when reconstructing the model and
        keeps the architecture consistent with the saved checkpoint.
        """
        super(GINConvNet, self).__init__()

        dim = 32
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.n_output = n_output

        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1; received {dropout}."
            )
        self.dropout_rate = dropout

        # Freezing state. These are runtime attributes and are not included in
        # the checkpoint state_dict.
        self._freeze_mode = "None"
        self._head_freeze_mode = "None"
        self._drug_encoder_frozen = False
        self._cell_encoder_frozen = False

        # ------------------------------------------------------------------
        # Drug encoder
        # ------------------------------------------------------------------
        nn1 = Sequential(
            Linear(num_features_xd, dim),
            ReLU(),
            Linear(dim, dim),
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        nn2 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim),
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        nn3 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim),
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(dim)

        nn4 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim),
        )
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(dim)

        nn5 = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, dim),
        )
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # ------------------------------------------------------------------
        # Cell-line encoder
        # ------------------------------------------------------------------
        # Retained for checkpoint compatibility with the original model,
        # although the current forward method does not use embedding_xt.
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        self.conv_xt_1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=8,
        )
        self.pool_xt_1 = nn.MaxPool1d(3)

        self.conv_xt_2 = nn.Conv1d(
            in_channels=n_filters,
            out_channels=n_filters * 2,
            kernel_size=8,
        )
        self.pool_xt_2 = nn.MaxPool1d(3)

        self.conv_xt_3 = nn.Conv1d(
            in_channels=n_filters * 2,
            out_channels=n_filters * 4,
            kernel_size=8,
        )
        self.pool_xt_3 = nn.MaxPool1d(3)

        if in_dim is not None:
            self.in_dim = in_dim
        else:
            with torch.no_grad():
                dummy_target = torch.zeros(1, 1, num_genes)
                dummy_target = dummy_target.to(next(self.parameters()).device)

                conv_xt = self.conv_xt_1(dummy_target)
                conv_xt = F.relu(conv_xt)
                conv_xt = self.pool_xt_1(conv_xt)

                conv_xt = self.conv_xt_2(conv_xt)
                conv_xt = F.relu(conv_xt)
                conv_xt = self.pool_xt_2(conv_xt)

                conv_xt = self.conv_xt_3(conv_xt)
                conv_xt = F.relu(conv_xt)
                conv_xt = self.pool_xt_3(conv_xt)

                self.in_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(self.in_dim, output_dim)

        # ------------------------------------------------------------------
        # Prediction head
        # ------------------------------------------------------------------
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)
        self.dropout = nn.Dropout(self.dropout_rate)

    # ----------------------------------------------------------------------
    # Layer-freezing support
    # ----------------------------------------------------------------------
    def _drug_encoder_modules(self):
        return (
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.conv3,
            self.bn3,
            self.conv4,
            self.bn4,
            self.conv5,
            self.bn5,
            self.fc1_xd,
        )

    def _cell_encoder_modules(self):
        return (
            self.embedding_xt,
            self.conv_xt_1,
            self.conv_xt_2,
            self.conv_xt_3,
            self.fc1_xt,
        )

    def _prediction_head_modules(self):
        return (
            self.fc1,
            self.fc2,
            self.out,
        )

    @staticmethod
    def _set_requires_grad(modules, requires_grad):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    @staticmethod
    def _parameters_from_modules(modules):
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]

    def _enforce_frozen_encoder_eval_mode(self):
        """
        Keep frozen encoders in evaluation mode.

        This is important for the drug encoder because BatchNorm running
        statistics would otherwise continue changing after requires_grad=False.
        """
        if self._drug_encoder_frozen:
            for module in self._drug_encoder_modules():
                module.eval()

        if self._cell_encoder_frozen:
            for module in self._cell_encoder_modules():
                module.eval()

    def _verify_freezing_configuration(self):
        drug_parameters = self._parameters_from_modules(
            self._drug_encoder_modules()
        )
        cell_parameters = self._parameters_from_modules(
            self._cell_encoder_modules()
        )

        expected_drug_trainable = not self._drug_encoder_frozen
        expected_cell_trainable = not self._cell_encoder_frozen

        if not all(
            parameter.requires_grad == expected_drug_trainable
            for parameter in drug_parameters
        ):
            raise RuntimeError(
                "Drug encoder freezing configuration is inconsistent."
            )

        if not all(
            parameter.requires_grad == expected_cell_trainable
            for parameter in cell_parameters
        ):
            raise RuntimeError(
                "Cell encoder freezing configuration is inconsistent."
            )

        expected_head_trainable = {
            "None": {
                "fc1": True,
                "fc2": True,
                "out": True,
            },
            "Freeze FC1": {
                "fc1": False,
                "fc2": True,
                "out": True,
            },
            "Freeze FC1 FC2": {
                "fc1": False,
                "fc2": False,
                "out": True,
            },
            "Freeze Head": {
                "fc1": False,
                "fc2": False,
                "out": False,
            },
        }[self._head_freeze_mode]

        for module_name, module in (
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("out", self.out),
        ):
            expected = expected_head_trainable[module_name]
            if not all(
                parameter.requires_grad == expected
                for parameter in module.parameters()
            ):
                raise RuntimeError(
                    "Prediction-head freezing configuration is inconsistent "
                    f"for {module_name}."
                )

        if self._drug_encoder_frozen:
            frozen_bn_modules = (
                self.bn1,
                self.bn2,
                self.bn3,
                self.bn4,
                self.bn5,
            )
            if any(module.training for module in frozen_bn_modules):
                raise RuntimeError(
                    "A frozen drug-encoder BatchNorm layer is still in "
                    "training mode."
                )

    def print_freezing_summary(self):
        total_parameters = sum(
            parameter.numel() for parameter in self.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        frozen_parameters = total_parameters - trainable_parameters

        def module_state(module):
            parameters = list(module.parameters())
            if not parameters:
                return "NO PARAMETERS"
            return (
                "TRAINABLE"
                if all(parameter.requires_grad for parameter in parameters)
                else "FROZEN"
            )

        print("\n" + "=" * 72)
        print("[Layer freezing] GraphDRP GIN configuration")
        print(f"[Layer freezing] Encoder mode: {self._freeze_mode}")
        print(f"[Layer freezing] Head mode: {self._head_freeze_mode}")
        print(
            "[Layer freezing] Drug encoder: "
            + ("FROZEN" if self._drug_encoder_frozen else "TRAINABLE")
        )
        print(
            "[Layer freezing] Cell encoder: "
            + ("FROZEN" if self._cell_encoder_frozen else "TRAINABLE")
        )
        print(
            f"[Layer freezing] Head fc1: {module_state(self.fc1)}"
        )
        print(
            f"[Layer freezing] Head fc2: {module_state(self.fc2)}"
        )
        print(
            f"[Layer freezing] Head out: {module_state(self.out)}"
        )
        print(
            f"[Layer freezing] Trainable parameters: "
            f"{trainable_parameters:,} / {total_parameters:,}"
        )
        print(
            f"[Layer freezing] Frozen parameters: {frozen_parameters:,}"
        )

        print("[Layer freezing] Parameter tensors:")
        for name, parameter in self.named_parameters():
            state = "TRAINABLE" if parameter.requires_grad else "FROZEN"
            print(
                f"  {state:9s}  {name:45s}  "
                f"shape={tuple(parameter.shape)}"
            )

        if self._drug_encoder_frozen:
            bn_eval = all(
                not module.training
                for module in (
                    self.bn1,
                    self.bn2,
                    self.bn3,
                    self.bn4,
                    self.bn5,
                )
            )
            print(
                "[Layer freezing] Frozen drug BatchNorm layers in eval mode: "
                f"{bn_eval}"
            )

        print("=" * 72 + "\n", flush=True)

    def configure_layer_freezing(
        self,
        freeze_mode="None",
        head_freeze_mode="None",
        verbose=True,
    ):
        """
        Configure trainable and frozen parts of the model.

        Encoder modes
        -------------
        None
            Train both encoders.
        Freeze Drug
            Freeze the GIN drug encoder.
        Freeze Cell
            Freeze the cancer-cell encoder.
        Freeze Both
            Freeze both encoders.

        Prediction-head modes
        ---------------------
        None
            Train fc1, fc2, and out (394,497 parameters with defaults).
        Freeze FC1
            Freeze fc1; train fc2 and out (131,329 parameters).
        Freeze FC1 FC2
            Freeze fc1 and fc2; train only out (129 parameters).
        Freeze Head
            Freeze fc1, fc2, and out (0 head parameters trainable).
        """
        freeze_mode = str(freeze_mode).strip()
        head_freeze_mode = str(head_freeze_mode).strip()

        if freeze_mode not in self.FREEZE_MODES:
            raise ValueError(
                f"Invalid freeze_mode={freeze_mode!r}. "
                f"Expected one of: {list(self.FREEZE_MODES)}"
            )

        if head_freeze_mode not in self.HEAD_FREEZE_MODES:
            raise ValueError(
                f"Invalid head_freeze_mode={head_freeze_mode!r}. "
                f"Expected one of: {list(self.HEAD_FREEZE_MODES)}"
            )

        # Reset everything first so this method can safely be called more
        # than once on the same model.
        for parameter in self.parameters():
            parameter.requires_grad = True

        self._freeze_mode = freeze_mode
        self._head_freeze_mode = head_freeze_mode

        self._drug_encoder_frozen = freeze_mode in (
            "Freeze Drug",
            "Freeze Both",
        )
        self._cell_encoder_frozen = freeze_mode in (
            "Freeze Cell",
            "Freeze Both",
        )

        if self._drug_encoder_frozen:
            self._set_requires_grad(
                self._drug_encoder_modules(),
                requires_grad=False,
            )

        if self._cell_encoder_frozen:
            self._set_requires_grad(
                self._cell_encoder_modules(),
                requires_grad=False,
            )

        # Head freezing is cumulative from the input side of the head.
        if head_freeze_mode in (
            "Freeze FC1",
            "Freeze FC1 FC2",
            "Freeze Head",
        ):
            self._set_requires_grad(
                (self.fc1,),
                requires_grad=False,
            )

        if head_freeze_mode in (
            "Freeze FC1 FC2",
            "Freeze Head",
        ):
            self._set_requires_grad(
                (self.fc2,),
                requires_grad=False,
            )

        if head_freeze_mode == "Freeze Head":
            self._set_requires_grad(
                (self.out,),
                requires_grad=False,
            )

        self._enforce_frozen_encoder_eval_mode()
        self._verify_freezing_configuration()

        if verbose:
            self.print_freezing_summary()

    def train(self, mode=True):
        """
        Preserve eval mode for frozen encoders whenever model.train() is called.
        """
        super().train(mode)

        if mode:
            self._enforce_frozen_encoder_eval_mode()

        return self

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Drug encoder
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)

        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))

        # Do not inject new dropout noise into a frozen drug representation.
        x = F.dropout(
            x,
            p=0.2,
            training=self.training and not self._drug_encoder_frozen,
        )

        # Cell-line encoder
        target = data.target
        target = target[:, None, :]

        conv_xt = self.conv_xt_1(target)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)

        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)

        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        xt = conv_xt.view(-1, self.in_dim)
        xt = self.fc1_xt(xt)

        # Prediction head
        xc = torch.cat((x, xt), 1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

###################
        # raw_out = self.out(xc)
        # out = torch.sigmoid(raw_out)

        # print(
        #     "raw_out stats:",
        #     raw_out.min().item(),
        #     raw_out.max().item(),
        #     raw_out.mean().item(),
        #     raw_out.std().item(),
        #     flush=True,
        # )

        # print(
        #     "sigmoid_out stats:",
        #     out.min().item(),
        #     out.max().item(),
        #     out.mean().item(),
        #     out.std().item(),
        #     flush=True,
        # )


        ##################

        out = self.out(xc)
        out = torch.sigmoid(out)

        return out, x