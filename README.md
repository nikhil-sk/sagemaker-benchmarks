# Sage Maker Containers Benchmarks

##Manual steps to start
###Raise SM limits
Limits for P3.16 and C5.18 should be raise. Climate-shlimate... 

###Create a bucket with datasources

Something like:
* mxnet-bln-data-sagemaker/imagenet/raw/train-480px
* mxnet-bln-data-sagemaker/imagenet/raw/validation-480px - imagenet validation set
* mxnet-bln-data-sagemaker/imagenet/processed/validation-480px - idx/rec files with imagenet
* mxnet-bln-data-sagemaker/small - small file just to make SM happy for the synthetic training 
