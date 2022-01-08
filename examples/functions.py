import torch
from tqdm.notebook import tqdm


def _run_training_loop(in_training_dataset, in_flow_model, in_optimizer, in_device):
    in_flow_model.train()
    loss_sum = 0
    n = len(in_training_dataset)
    for x in in_training_dataset:
        x = x.to(in_device)
        in_optimizer.zero_grad()
        loss = in_flow_model.nll_mean(x)
        loss.backward()
        in_optimizer.step()
        loss_sum += loss.item()
    return loss_sum / n


def _run_validation_loop(in_validation_dataset, in_flow_model, in_device):
    in_flow_model.eval()
    loss_sum = 0
    n = len(in_validation_dataset)
    with torch.no_grad():
        for x in in_validation_dataset:
            x = x.to(in_device)
            loss = in_flow_model.nll_mean(x)
            loss_sum += loss.item()
    return loss_sum / n


def run_training(in_n_epoch, in_training_dataset, in_validation_dataset, in_flow_model, in_optimizer, in_device):
    print("Starting Training Loop")
    for epoch in tqdm(range(in_n_epoch)):
        loss_training = _run_training_loop(in_training_dataset, in_flow_model, in_optimizer, in_device)
        loss_validation = _run_validation_loop(in_validation_dataset, in_flow_model, in_device)
        print(f"End Epoch with training loss:{loss_training} and validtion loss:{loss_validation}")
