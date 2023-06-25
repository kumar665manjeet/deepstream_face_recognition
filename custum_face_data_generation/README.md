This directory contains jupyter notbook to get the specific format .npz face encoading data.

Please create vertual env using conda  with python 3.6 because here I have used facenet model with pretrain on older tensorflow and keras versions after that onlyrun the jupyter notebook.



* Convert facenet model to TensorRT engine using this jupyter notebook. The steps in the jupyter notebook is taken from Nvidia official tutorial.
* when converting pb file to onnx use below command instead: python -m tf2onnx.convert --input facenet.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output facenet.onnx Note: make sure to use this command --inputs-as-nchw input_1:0 while converting to ONNX to avoid having this error: Error in NvDsInferContextImpl::preparePreprocess() <nvdsinfer_context_impl.cpp:874> [UID = 2]: RGB/BGR input format specified but network input channels is not 3

