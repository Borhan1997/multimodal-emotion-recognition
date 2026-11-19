import os

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    # Configure via environment variables (see .env.example)
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket = os.environ["S3_BUCKET"]

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{bucket}/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=role_arn,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25,
            "num-workers": 4
        },
        metric_definitions=[
            {"Name": n, "Regex": rf'"Name": "{n}", "Value": ([0-9.eE+-]+)'}
            for n in ["train_loss", "val_loss",
                      "val_emotion_precision", "val_emotion_accuracy",
                      "val_sentiment_precision", "val_sentiment_accuracy",
                      "test_loss", "test_emotion_precision", "test_emotion_accuracy",
                      "test_sentiment_precision", "test_sentiment_accuracy"]
        ],
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": f"s3://{bucket}/dataset/train",
        "validation": f"s3://{bucket}/dataset/dev",
        "test": f"s3://{bucket}/dataset/test"
    })


if __name__ == "__main__":
    start_training()