import wandb
import random

wandb.login()

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # Simulate a training run
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})