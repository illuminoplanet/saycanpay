import os
import pickle
import random
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl


@dataclass
class Sample:
    graph_id: int
    goal: str
    actions: list


@dataclass
class CanSample:
    context: str
    pos_action: str
    neg_action: str
    neg_sample_action: str


def load_dataset(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    samples = []
    for task in data:
        timestep = 1
        plan = []
        for sample in data[task]:
            if sample["timesteps"] == timestep:
                plan.append(sample["action"])
            else:
                samples.append(Sample(sample["env_id"], sample["instruction"], plan))
                timestep = 1
                plan = [sample["action"]]
            timestep += 1

    print(f"\n[Dataset] Loaded {len(samples)} samples")
    return samples


class CanDataset(Dataset):
    def __init__(self, file_path):
        samples = load_dataset(file_path)
        self.can_samples = self.process(samples)

    def __len__(self):
        return len(self.can_samples)

    def __getitem__(self, item):
        return self.can_samples[item]

    def process(self, samples):
        can_samples = []
        print("\n[Can] Processing samples for Can")
        for sample in tqdm(samples):
            context = ["[Goal]", sample.goal]
            for i in range(len(sample.actions)):
                pos_action = sample.actions[i]

                neg_actions = []
                for action in sample.actions:
                    if action != pos_action:
                        neg_actions.append(action)
                neg_action = np.random.choice(neg_actions)
                assert (
                    neg_action is not None
                ), "Negative action should not be None, double check if # actions in the sample are at least 2"

                pos_action = f"{pos_action}"
                neg_action = f"{neg_action}"

                num_neg_samples = 2
                for _ in range(num_neg_samples):
                    neg_sample = random.choice([s for s in samples if s != sample])
                    neg_sample_action = random.choice(neg_sample.actions)
                    discard_samples = [sample]

                    while (
                        neg_sample_action == pos_action
                        or neg_sample.goal == sample.goal
                    ):
                        discard_samples.append(neg_sample)
                        neg_sample = random.choice(
                            [s for s in samples if s not in discard_samples]
                        )
                        neg_sample_action = random.choice(neg_sample.actions)

                    neg_sample_action = f"{neg_sample_action}"
                    can_samples.append(
                        vars(
                            CanSample(
                                " ".join(context),
                                pos_action,
                                neg_action,
                                neg_sample_action,
                            )
                        )
                    )

                if i < len(sample.actions) - 1:
                    context.append(f"[Step {i + 1}] {pos_action}")

        print(f"\n[Can] Processed {len(can_samples)} Can samples")
        return can_samples


@dataclass
class PaySample:
    context: str
    action: str
    dist: float


class PayDataset(Dataset):
    def __init__(self, file_path):
        samples = load_dataset(file_path)
        self.pay_samples = self.process(samples)

    def __len__(self):
        return len(self.pay_samples)

    def __getitem__(self, item):
        return self.pay_samples[item]

    def process(self, samples):
        pay_samples = []
        print("\n[Pay] Processing samples for Pay")
        for sample in samples:
            context = ["[Goal]", sample.goal]
            for i in range(len(sample.actions)):
                gamma = 1.5
                final_reward = 1
                dist = np.float32(
                    1 / (gamma ** (len(sample.actions) - i)) * final_reward
                )
                pay_samples.append(
                    vars(
                        PaySample(
                            context=" ".join(context),
                            action=f"[Step {i + 1}] {sample.actions[i]}",
                            dist=dist,
                        )
                    )
                )
                neg_actions = []
                for action in sample.actions:
                    if action != sample.actions[i]:
                        neg_actions.append(action)

                num_neg_actions = 1
                for _ in range(num_neg_actions):
                    neg_action = np.random.choice(neg_actions)
                    pay_samples.append(
                        vars(
                            PaySample(
                                context=" ".join(context),
                                action=f"[Step {i + 1}] {neg_action}",
                                dist=np.float32(0),
                            )
                        )
                    )
                    neg_actions.remove(neg_action)

                context.append(f"[Step {i + 1}] {sample.actions[i]}")

        print(f"\n[Pay] Processed {len(pay_samples)} Pay samples")
        return pay_samples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model,
        file_path,
        batch_size,
        train_val_split,
        completed_action,
    ):
        super().__init__()
        self.val_set = None
        self.train_set = None
        self.model = model
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.completed_action = completed_action

    def prepare_data(self) -> None:
        if self.model == "can":
            self.dataset = CanDataset(self.file_path)
        elif self.model == "pay":
            self.dataset = PayDataset(self.file_path)

    def setup(self, stage: str) -> None:
        train_size = int(self.train_val_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=10)
