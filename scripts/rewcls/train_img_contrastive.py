import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from diwa.dataset.rewcls_dataset import RewClsDataset
from diwa.model.rewcls.resnet_contrastive import ResNetClassifier
import wandb


# SimCLR-like loss: NT-Xent Loss (with ground truth labels for positive pairs)
def nt_xent_loss(z, labels, temperature=0.5):
    batch_size = z.shape[0]

    # Normalize the representations
    z = torch.nn.functional.normalize(z, dim=1)

    # Cosine similarity matrix between all elements in the batch
    similarity_matrix = torch.matmul(z, z.T)

    # Scale the similarity matrix with the temperature
    similarity_matrix = similarity_matrix / temperature

    # Create a mask for positive pairs based on the ground truth labels
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(similarity_matrix.device)

    # Mask out the diagonal (self-contrast, i.e., a sample shouldn't compare to itself)
    mask = torch.eye(batch_size, dtype=torch.bool).to(similarity_matrix.device)

    # For stability: subtract the maximum value for each row before exponentiating
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Calculate softmax scores (exp/log sums) for the denominator, excluding self-similarities
    exp_logits = torch.exp(logits) * (~mask)  # Zero out diagonal instead of -inf
    exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

    # Calculate log-probabilities for positive pairs
    log_prob = logits - torch.log(exp_logits_sum + 1e-10)

    # Select the log-probabilities of the positive pairs (where positive_mask == 1)
    positive_log_prob = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-10)

    # Loss is the mean negative log likelihood of the positive pairs
    loss = -positive_log_prob.mean()

    return loss


def train_model(cfg):
    train_dataset = RewClsDataset(cfg.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    if cfg.val_data_path:
        val_dataset = RewClsDataset(cfg.val_data_path)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    else:
        val_dataset = None
        val_dataloader = None

    model = ResNetClassifier()
    model.to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.num_epochs):
        total_loss = 0
        train_contrastive_loss = 0
        train_ce_loss = 0
        true_y = []
        pred_y = []
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)
            optimizer.zero_grad()
            z = model.encode(X_batch)
            contrastive_loss = nt_xent_loss(z, y_batch, temperature=cfg.temperature)
            outputs = model.classify(z)
            ce_loss = ce_criterion(outputs, y_batch)

            loss = contrastive_loss + cfg.alpha * ce_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_contrastive_loss += contrastive_loss.item()
            train_ce_loss += ce_loss.item()
            true_y.extend(y_batch.cpu().numpy())
            pred_y.extend(outputs.argmax(dim=1).cpu().numpy())

        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        train_precision = precision_score(true_y, pred_y, pos_label=1, average="binary")
        train_recall = recall_score(true_y, pred_y, pos_label=1, average="binary")

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Train Loss: {total_loss/len(train_dataloader):.5f}")
        print(f"Train Precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}")
        if cfg.wandb:
            wandb.log(
                {
                    "train_loss": total_loss / len(train_dataloader),
                    "train_contrastive_loss": train_contrastive_loss / len(train_dataloader),
                    "train_ce_loss": train_ce_loss / len(train_dataloader),
                    "train_epoch": epoch,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                }
            )

        train_precision = precision_score(true_y, pred_y, pos_label=0, average="binary")
        train_recall = recall_score(true_y, pred_y, pos_label=0, average="binary")
        if cfg.wandb:
            wandb.log({"train_precision_0": train_precision, "train_recall_0": train_recall})

        if epoch % 5 == 0 and val_dataloader is not None:
            val_loss = 0
            val_contrastive_loss = 0
            val_ce_loss = 0
            true_y = []
            pred_y = []
            with torch.no_grad():
                for X_batch, y_batch in val_dataloader:
                    X_batch = X_batch.to(cfg.device)
                    y_batch = y_batch.to(cfg.device)
                    z = model.encode(X_batch)
                    contrastive_loss = nt_xent_loss(z, y_batch, temperature=cfg.temperature)
                    outputs = model.classify(z)
                    ce_loss = ce_criterion(outputs, y_batch)
                    loss = contrastive_loss + cfg.alpha * ce_loss
                    val_loss += loss.item()
                    val_contrastive_loss += contrastive_loss.item()
                    val_ce_loss += ce_loss.item()
                    true_y.extend(y_batch.cpu().numpy())
                    pred_y.extend(outputs.argmax(dim=1).cpu().numpy())

            true_y = np.array(true_y)
            pred_y = np.array(pred_y)

            val_precision = precision_score(true_y, pred_y, pos_label=1, average="binary")
            val_recall = recall_score(true_y, pred_y, pos_label=1, average="binary")

            print(f"Validation Loss: {val_loss/len(val_dataloader):.5f}")
            print(f"Validation Precision: {val_precision:.2f}, Validation Recall: {val_recall:.2f}")
            if cfg.wandb:
                wandb.log(
                    {
                        "val_loss": val_loss / len(val_dataloader),
                        "val_contrastive_loss": val_contrastive_loss / len(val_dataloader),
                        "val_ce_loss": val_ce_loss / len(val_dataloader),
                        "val_epoch": epoch,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                    }
                )

            val_precision = precision_score(true_y, pred_y, pos_label=0, average="binary")
            val_recall = recall_score(true_y, pred_y, pos_label=0, average="binary")
            if cfg.wandb:
                wandb.log({"val_precision_0": val_precision, "val_recall_0": val_recall})
    print("Training complete.")

    torch.save(
        model.state_dict(),
        os.path.join(cfg.model_out_dir, cfg.model_save_name),
    )
    print("Model saved to", os.path.join(cfg.model_out_dir, cfg.model_save_name))


@hydra.main(version_base="1.3", config_path="../../config/rewcls", config_name="contrastive_img")
def main(cfg: DictConfig):
    train_data_path = Path(cfg.train_data_path)
    model_out_dir = Path(cfg.model_out_dir)

    assert train_data_path.exists(), f"Train data path {train_data_path} does not exist."

    if cfg.val_data_path:
        val_data_path = Path(cfg.val_data_path)
        assert val_data_path.exists(), f"Validation data path {val_data_path} does not exist."

    if not model_out_dir.exists():
        model_out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.run,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    train_model(cfg)

    # save config
    OmegaConf.save(
        cfg,
        os.path.join(cfg.model_out_dir, "config.yaml"),
    )


if __name__ == "__main__":
    main()
