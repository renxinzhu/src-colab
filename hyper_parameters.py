import torch


class ParametersBase:
    """ Just a base class allowing class attribute iteration for HyperParameter and GlobalSetting """

    def items(self):
        return [(a, getattr(self, a)) for a in dir(self) if (
            not a.startswith('__') and a not in (
                'items', 'pretty_print', 'to_string_log')
        )]

    def pretty_print(self):
        for key, value in self.items():
            print(f'{key}: {value}')

    def to_string_log(self):
        string_log = []
        for key, value in self.items():
            string_log.append(f'{key}: {value}')
        return '\n'.join(string_log)


class HyperParameters(ParametersBase):
    local_epochs = 1
    confidence_threshold = 0.75
    lr = 1e-3
    lr_factor = 3
    lr_patience = 5
    lambda_s = 10
    lambda_iccs = 1e-2
    lambda_l1 = 1e-4
    lambda_l2 = 10
    lambda_src = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # psi_factor = 0.2 #need to be added later


class GlobalSetting(ParametersBase):
    model = 'SmallModel'  # @param ['BigModel', 'SmallModel']
    num_clients = 100
    R = 0.05
    rounds = 200
    batch_size_s = 10
    batch_size_u = 100
    dataset_name = 'cifar10'
    label_ratio = 0.05
    iid = True
    h_interval = 10
    num_helpers = 2


hyper_parameters = HyperParameters()
setting = GlobalSetting()
