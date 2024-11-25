from pathlib import Path
from typing import Dict
from mads_hackathon import datasets, metrics
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
from mads_hackathon.models import CNNConfig as Config
import ray
import torch
import mlflow
from filelock import FileLock
from loguru import logger
from mads_hackathon.models import CNN
from dataclasses import asdict
import numpy as np
from mads_hackathon.metrics import caluclate_cfm
from mltrainer import ReportTypes, Trainer, TrainerSettings
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"  # type: ignore
    logger.info(f"Using {device}")
    if device != "cpu":
        logger.warning(
            f"using acceleration with {device}." "Check if it actually speeds up!"
        )
        
    data_dir = config["data_dir"]
    # gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    preprocessor = BasePreprocessor()
    trainfile = (data_dir / "heart_big_train.parq").resolve()
    validfile = (data_dir / "heart_big_valid.parq").resolve()
    trainfile.exists(), validfile.exists()
    
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=config["matrixshape"])
    validdataset = datasets.HeartDataset2D(validfile, target="target", shape=config["matrixshape"])
    traindataset.to(device)
    validdataset.to(device)
        
    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        trainstreamer = BaseDatastreamer(traindataset, preprocessor = preprocessor, batchsize=config["batchsize"])
        validstreamer = BaseDatastreamer(validdataset, preprocessor = preprocessor, batchsize=config["batchsize"])

    # we set up the metric
    # and create the model with the config
    accuracy = metrics.Accuracy()
    model = CNN(config)

    trainersettings = TrainerSettings(
        epochs=config["epochs"],
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(trainstreamer),  # type: ignore
        valid_steps=len(validstreamer),  # type: ignore
        reporttypes=[ReportTypes.RAY, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    # because we set reporttypes=[ReportTypes.RAY]
    # the trainloop wont try to report back to tensorboard,
    # but will report back with ray
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop.
    # This is why we set earlystop_kwargs=None, because we
    # are handing over this control to ray.

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam, # type: ignore
        traindataloader=trainstreamer.stream(),
        validdataloader=validstreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
        )
    
    # modify the tags when you change them!
    mlflow.set_tracking_uri("http://145.38.195.42:5002")
    mlflow.set_experiment("FirstHyperTest")
    mlflow.start_run()
    mlflow.set_tag("model", "CNN")
    mlflow.set_tag("dataset", "heart2D")
    mlflow.set_tag("dev", config["dev"])
    mlflow.log_param("scheduler", "None")
    mlflow.log_param("earlystop", "None")

    mlflow.log_params(config)
    mlflow.log_param("epochs", trainersettings.epochs)
    mlflow.log_param("matrix0", config["matrixshape"][0])
    mlflow.log_param("matrix1", config["matrixshape"][1])
    mlflow.log_param("optimizer", str(trainer.optimizer))
    mlflow.log_params(trainersettings.optimizer_kwargs)
    
    trainer.loop()
    cfm = caluclate_cfm(model, validstreamer)
    for i, tp in enumerate(np.diag(cfm)):
        mlflow.log_metric(f"TP_{i}", tp)
        
if __name__ == "__main__":
    NUM_SAMPLES = 5
    MAX_EPOCHS = 10
    ray.init()

    data_dir = Path("/home/ydibbet/mads-hackathon/hackathon-data").resolve()
    tune_dir = Path("models/ray").resolve()
    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    config = {
        "epochs": 10,
        "output_size": 20,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "dropout": tune.uniform(0.0, 0.3),
        "dev": "youri",
        "matrixshape": (4,48),
        "batchsize": 64,
        "input_channels": 1,
        "hidden": tune.randint(16, 128),
        "kernel_size": 3, # tune.randint(1, 12),
        "maxpool": 2, # tune.randint(1, 4),
        "num_layers": tune.randint(2, 5),
        "num_classes": 5, # tune.randint(2, 5),  
    }
    
    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
    )

    ray.shutdown()
