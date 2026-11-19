import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys

from install_ffmpeg import install_ffmpeg

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)  # NOTE: unused, LRs are set per-module in MultimodalTrainer
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N rows of each split (local smoke test)")
    parser.add_argument("--num-workers", type=int, default=0)

    # Data Directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():
    if not install_ffmpeg():
        print("Error : FFMPEG installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available Audio backends:")
    #print(str(torchaudio.list_audio_backends()))

    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers
    )

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"""Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training Video directory: {os.path.join(args.train_dir, 'train_splits')}""")

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.model_dir, "model.pth")

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log Validation metrics in SageMaker format
        print(json.dumps({
            "metrics" : [
                {"Name": "train_loss", "Value": train_loss["total"]},
                {"Name": "val_loss", "Value": val_loss["total"]},
                {"Name": "val_emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "val_emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "val_sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name": "val_sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]}
            ]
        }))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), best_model_path)

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    # Log Test metrics in SageMaker format
    print(json.dumps({
        "metrics" : [
            {"Name": "test_loss", "Value": test_loss["total"]},
            {"Name": "test_emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test_emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test_sentiment_precision", "Value": test_metrics["sentiment_precision"]},
            {"Name": "test_sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]}
        ]
    }))


if __name__ == "__main__":
    main()