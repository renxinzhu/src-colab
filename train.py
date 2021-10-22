import gc
import torch
from client import Client
from server import Server
from datasets import generate_dataloaders, generate_test_dataloader
from logger import Logger
from hyper_parameters import hyper_parameters, setting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}.')

dataloaders = generate_dataloaders(
    setting.dataset_name,
    setting.num_clients,
    setting.label_ratio,
    setting.iid,
    setting.batch_size_s,
    setting.batch_size_u
)
test_dataloader = generate_test_dataloader(setting.dataset_name)

clients = [Client(hyper_parameters, dataloader = dataloader,
    test_dataloader = test_dataloader, client_id = client_id) 
    for client_id,dataloader in enumerate(dataloaders)]

server = Server(hyper_parameters, test_dataloader = test_dataloader)

for r in range(setting.rounds):
    state_dict = server.state_dict()
    num_connected = torch.randint(0, setting.num_clients,
                          (int(setting.num_clients * setting.R),))
    server.logger.print(
        f'training clients (round:{r}, connected:{num_connected})')
    if r % setting.h_interval == 0:
        server.save_client_psis(clients)
        server.logger.print('tree constructed')

    for idx in num_connected:
        client = clients[idx]
        helper_psis = server.get_helpers_by_direct_calculation(
            setting.num_helpers, client.get_psi_tensors())

        client.local_train(state_dict, helper_psis)

    server.aggregate([clients[idx].state_dict() for idx in num_connected])

    test_acc,test_loss = server.evaluate()

    server.logger.print(
        f'aggr_acc:{round(test_acc,4)},aggr_loss:{round(test_loss,4)}')
    server.logger.save_current_state(
        {"round": r, "aggr_acc": test_acc, "aggr_loss": test_loss})

    gc.collect()
