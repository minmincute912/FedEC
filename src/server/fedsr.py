import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F

from src.server.fedavg import FedAvgServer
from src.client.fedsr import FedSRClient
from src.utils.models import DecoupledModel
from src.utils.constants import NUM_CLASSES
from src.utils.tools import trainable_params, NestedNamespace


def get_fedsr_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--L2R_coeff", type=float, default=1e-2)
    parser.add_argument("--CMI_coeff", type=float, default=5e-4)
    return parser.parse_args(args_list)


class FedSRModel(DecoupledModel):
    # modify base model to suit FedSR
    def __init__(self, base_model: DecoupledModel, dataset) -> None:
        super().__init__()
        self.z_dim = base_model.classifier.in_features
        out_dim = 2 * self.z_dim
        self.base = base_model.base
        self.map_layer = nn.Linear(self.z_dim, out_dim)
        self.classifier = base_model.classifier
        self.r_mu = nn.Parameter(torch.zeros(NUM_CLASSES[dataset], self.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(NUM_CLASSES[dataset], self.z_dim))
        self.C = nn.Parameter(torch.ones([]))

    def featurize(self, x, num_samples=1, return_dist=False):
        # designed for FedSR
        z_params = F.relu(self.map_layer(F.relu(self.base(x))))
        z_mu = z_params[:, : self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim :])
        z_dist = distrib.Independent(distrib.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        z = self.featurize(x)
        logits = self.classifier(z)
        return logits


class FedSRServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedSR",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        # reload the model
        self.model = FedSRModel(self.model, self.args.common.dataset)
        self.model.check_avaliability()

        _init_global_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        _init_personal_params = OrderedDict(self.model.named_buffers())
        self.global_model_params = OrderedDict(
            zip(self.trainable_params_name, _init_global_params)
        )
        self.clients_personal_model_params = {
            client_id: {
                key: param.cpu().clone() for key, param in _init_personal_params.items()
            }
            for client_id in range(self.client_num)
        }

        if self.args.common.external_model_params_file and os.path.isfile(
            self.args.common.external_model_params_file
        ):
            # load pretrained params
            self.global_model_params = torch.load(
                self.args.common.external_model_params_file, map_location=self.device
            )

        self.init_trainer(FedSRClient)
