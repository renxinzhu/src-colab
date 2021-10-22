from typing import Dict, cast
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from base_model import Backbone, Model
from hyper_parameters import HyperParameters
from loss import iccs_loss, regularization_loss, src_loss
from custom_types import IStateDict
from typing import List, cast
from util import replace_psi_in_state_dict


class Client(Backbone):

    def local_train(
        self,
        state_dict: IStateDict,
        helper_psis: List[List[torch.Tensor]],
    ):
        self.load_state_dict(state_dict)
        self.helper_psis = helper_psis

        self._train_supervised()
        self._train_unsupervised()

        SHOW_LOCAL_EVAL = True #enable it to display local acc/loss, disable it for faster training. will be removed later.
        if SHOW_LOCAL_EVAL:
            test_acc, test_loss = self.evaluate()
            n_train_s = len(self.dataloader['labeled'].dataset)
            n_train_u = len(self.dataloader['unlabeled'].dataset)
            self.logger.print(
                f"test_acc:{round(test_acc,4)},test_loss:{round(test_loss,4)},lr:{self.lr},\
            n_train_s:{n_train_s},n_train_u:{n_train_u},n_conf:{self.n_conf}")
            self.adjust_lr(test_acc)
    
    def _train_supervised(self):
        opt = torch.optim.SGD(self.get_sigma_parameters(
            True), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        device = self.hyper_parameters.device

        for _ in range(self.hyper_parameters.local_epochs):
            for __, X, y in self.dataloader['labeled']:
                self.train()
                X = X.to(device)
                y = y.to(device)

                pred = self.forward(X)
                loss = loss_fn(pred, y) * self.hyper_parameters.lambda_s

                opt.zero_grad()
                loss.backward()
                opt.step()

    def _train_unsupervised(self):
        opt = torch.optim.SGD(self.get_psi_parameters(
            True), lr=self.lr)
        device = self.hyper_parameters.device
        confident_counts = 0

        for _ in range(self.hyper_parameters.local_epochs):
            for noised_X, X, __ in self.dataloader['unlabeled']:
                self.train()
                X = X.to(device)
                noised_X = noised_X.to(device)

                last_feature_map, output = self.forward(X, True)
                pred = F.softmax(output, dim=-1)

                max_values = cast(torch.Tensor, torch.max(pred, dim=1).values)
                confident_idxes = [idx for idx, value in enumerate(max_values.tolist()) if (
                    value > self.hyper_parameters.confidence_threshold
                )]

                confident_counts += len(confident_idxes)

                loss = regularization_loss(
                    self.get_sigma_parameters(False),
                    self.get_psi_parameters(False),
                    self.hyper_parameters.lambda_l1,
                    self.hyper_parameters.lambda_l2,
                )

                if confident_idxes:
                    confident_pred = pred[confident_idxes]
                    confiden_feature_map = last_feature_map[confident_idxes]
                    confident_noised_X = noised_X[confident_idxes]
                    confident_X = X[confident_idxes]

                    noised_pred = self.forward(
                        confident_noised_X)  # type: torch.Tensor
                    mean_feature_map, helper_preds = self._helper_predictions(
                        confident_X)

                    loss += iccs_loss(confident_pred,
                                      noised_pred, helper_preds, self.hyper_parameters.lambda_iccs)

                    loss += src_loss(confiden_feature_map, mean_feature_map,
                                     self.hyper_parameters.lambda_src)

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.n_conf = confident_counts

    def _helper_predictions(self, X: torch.Tensor):
        '''make prediction one by one instead of creating all in the RAM to reduce RAM usage'''
        with torch.no_grad():
            helpers_pred = []
            helper_feature_maps = []
            for helper_psi in self.helper_psis:
                model = Model().to(self.hyper_parameters.device)

                state_dict = replace_psi_in_state_dict(
                    self.state_dict(), helper_psi)
                model.load_state_dict(state_dict)

                feature_map, pred = model(X, last_feature_map=True)
                helpers_pred.append(pred)
                helper_feature_maps.append(feature_map)
            return torch.mean(torch.stack(helper_feature_maps), dim=0), helpers_pred
