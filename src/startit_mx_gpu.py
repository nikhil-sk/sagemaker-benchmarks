import sagemaker
from sagemaker.mxnet import MXNet

sagemaker_session = sagemaker.Session()

tf_estimator = MXNet(
    sagemaker_session=sagemaker_session,
    entry_point="smtrain.py",
    source_dir="../benchmarks/tr-gpu/mx",
    role="SageMakerRole",
    train_instance_count=12,
    train_instance_type="ml.p3.16xlarge",
    image_name="841569659894.dkr.ecr.us-east-1.amazonaws.com/beta-mxnet-training:1.4.1-py3-gpu-build",
    py_version="py3",
    output_path="s3://bai-results-sagemaker",
    train_volume_size=200,
    framework_version="1.4",
    distributions={"parameter_server": {"enabled": True}},
)

data = {
    #"s1": "s3://mxnet-bln-data-sagemaker/small"
    "train": "s3://mxnet-asimov-data-sagemaker/imagenet/processed/train-480px-q95.rec",
    "trainidx": "s3://mxnet-asimov-data-sagemaker/imagenet/processed/train-480px-q95.idx",
    "validate": "s3://mxnet-asimov-data-sagemaker/imagenet/processed/val-480px-q95.rec",
    "validx": "s3://mxnet-asimov-data-sagemaker/imagenet/processed/val-480px-q95.idx",
}

tf_estimator.fit(data, logs=True, wait=True)
