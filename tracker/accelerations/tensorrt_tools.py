'''
convert models to tensorrt engine and inference
'''

import os
import torch
import tensorrt as trt
from loguru import logger

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np 

from collections import OrderedDict, namedtuple

from loguru import logger 

class TensorRTConverter(object):

    def __init__(self, model, input_shape, ckpt_path, min_opt_max_batch=[1, 1, 1], use_fp16=False, 
                device=torch.device('cpu'), load_ckpt=False, ckpt_key='state_dict'):
        """
        Args:
            model: torch.nn.Module, model
            input_shape: List[int], (c, h, w), input tensor shape
            ckpt_path: str, used to save the onnx and tensorrt file
            use_fp16: bool, whether use fp16
            device: str | torch.device, cpu or gpu
            load_ckpt: bool, whether load the pt/pth checkpoint file 
            ckpt_key: str, the weight key in ckpt dict
        """
        self.model = model         
        self.input_shape = input_shape

        postfix_length = len(ckpt_path.split('.')[-1])  # .pt/.pth
        self.onnx_model = ckpt_path[:-postfix_length] + 'onnx'
        self.trt_model = ckpt_path[:-postfix_length] + 'engine'
        self.use_fp16 = use_fp16

        self.device = device 

        if load_ckpt:
            logger.info(f'to convert TensorRT, load ckpt {ckpt_path} first')
            self.load_ckpt(ckpt_path, ckpt_key)
            logger.info('load ckpt done')
        
        
        self.min_input_shape = tuple([min_opt_max_batch[0], *input_shape])
        self.opt_input_shape = tuple([min_opt_max_batch[1], *input_shape])
        self.max_input_shape = tuple([min_opt_max_batch[2], *input_shape])

        # check tensor rt version
        self.is_trt_10 = int(trt.__version__.split(".")[0]) >= 10

        # constants
        self.WORK_SPACE = 1 << 30
        self.INPUT_NAME = "images"
        self.OUTPUT_NAME = "output"
        self.OPSET_VERSION = 12

    def load_ckpt(self, ckpt_path, ckpt_key):
        """
        load checkpoint file if needed
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get(ckpt_key, ckpt)
        new_state = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[len("module."):]
            if name.startswith("model."):
                name = name[len("model."):]
            new_state[name] = v
        self.model.load_state_dict(new_state)
        self.model.eval()

    def export_onnx(self):
        logger.info('to convert TensorRT, convert onnx first')

        if os.path.exists(self.onnx_model):
            logger.warning(f'the onnx {self.onnx_model} already exists, so the export progress is stopped')
            return 

        dummy = torch.randn([1, *self.input_shape]).cuda()
        torch.onnx.export(
            self.model, dummy, self.onnx_model,
            input_names=[self.INPUT_NAME],
            output_names=[self.OUTPUT_NAME],
            opset_version=self.OPSET_VERSION,
            do_constant_folding=True,
            dynamic_axes={self.INPUT_NAME:{0:"batch_size"}, self.OUTPUT_NAME:{0:"batch_size"}}
        )
        logger.info(f'convert onnx done, save path: {self.onnx_model}')

    def export(self):

        if os.path.exists(self.trt_model):
            logger.warning(f'the engine {self.trt_model} already exists, so the export progress is stopped')
            return 

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        # convert onnx
        self.export_onnx()  

        logger.info('converting tensorrt')
        with open(self.onnx_model, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    trt_logger.error(parser.get_error(error))

        # build configs
        config = builder.create_builder_config()
        if self.use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16) 
            logger.info('enabled fp16')

        if self.is_trt_10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.WORK_SPACE)
        else:
            config.max_workspace_size = self.WORK_SPACE
        profile = builder.create_optimization_profile()
        # support dynamic batch
        profile.set_shape(
            self.INPUT_NAME, 
            min=self.min_input_shape,  
            opt=self.opt_input_shape, 
            max=self.max_input_shape
        )
        config.add_optimization_profile(profile)

        logger.info('saving tensorrt engine')
        if self.is_trt_10:
            engine = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_engine(network, config)

        with open(self.trt_model, "wb") as f:
            f.write(engine if self.is_trt_10 else engine.serialize())

        logger.info(f'convert tensorrt done, save path: {self.trt_model}')


class TensorRTInference(object):

    def __init__(self, engine_path, min_opt_max_batch=[1, 1, 1], use_fp16=False, 
                device=torch.device('cpu')):
        

        self.use_fp16 = use_fp16
        self.engine_path = engine_path
        self.device = device

        self.is_trt_10 = int(trt.__version__.split(".")[0]) >= 10

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # load engine
        logger.info(f'loading tensor rt engine {engine_path}')
        self.load_engine()
        logger.info(f'load tensor rt engine done')
        
        # constants
        self.INPUT_NAME = "images"
        self.OUTPUT_NAME = "output"

    def load_engine(self):
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        # Deserialize the engine
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        # create context
        self.context = self.engine.create_execution_context()

        # Execution context
        self.bindings = OrderedDict()

        num = range(self.engine.num_io_tensors) if self.is_trt_10 else range(self.engine.num_bindings)

        # Parse bindings
        for index in num:
            if self.is_trt_10:
                name = self.engine.get_tensor_name(index)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input and -1 in tuple(self.engine.get_tensor_shape(name)):
                        self.context.set_input_shape(name, tuple(self.engine.get_tensor_profile_shape(name, 0)[1]))

                shape = tuple(self.context.get_tensor_shape(name))

            else:
                name = self.engine.get_binding_name(index)
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                is_input = self.engine.binding_is_input(index)

                # Handle dynamic shapes
                if is_input and -1 in self.engine.get_binding_shape(index):
                    profile_index = 0
                    min_shape, opt_shape, max_shape = self.engine.get_profile_shape(profile_index, index)
                    self.context.set_binding_shape(index, opt_shape)

                shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def inference(self, input_tensor):
        """
        Args:
            input_tensor: torch.Tensor, shape (b, c, h, w)
        """
       
        temp_im_batch = input_tensor.clone()
        batch_array = []
        inp_batch = input_tensor.shape[0]
        out_batch = self.bindings[self.OUTPUT_NAME].shape[0]
        resultant_features = []

        # Divide batch to sub batches
        while inp_batch > out_batch:
            batch_array.append(temp_im_batch[:out_batch])
            temp_im_batch = temp_im_batch[out_batch:]
            inp_batch = temp_im_batch.shape[0]
        if temp_im_batch.shape[0] > 0:
            batch_array.append(temp_im_batch)

        for temp_batch in batch_array:
            # Adjust for dynamic shapes
            if temp_batch.shape != self.bindings[self.INPUT_NAME].shape:
                if self.is_trt_10:

                    self.context.set_input_shape(self.INPUT_NAME, temp_batch.shape)
                    self.bindings[self.INPUT_NAME] = self.bindings[self.INPUT_NAME]._replace(shape=temp_batch.shape)
                    self.bindings[self.OUTPUT_NAME].data.resize_(tuple(self.context.get_tensor_shape(self.OUTPUT_NAME)))
                else:
                    i_in = self.model_.get_binding_index(self.INPUT_NAME)
                    i_out = self.model_.get_binding_index(self.OUTPUT_NAME)
                    self.context.set_binding_shape(i_in, temp_batch.shape)
                    self.bindings[self.INPUT_NAME] = self.bindings[self.INPUT_NAME]._replace(shape=temp_batch.shape)
                    output_shape = tuple(self.context.get_binding_shape(i_out))
                    self.bindings[self.OUTPUT_NAME].data.resize_(output_shape)

            s = self.bindings[self.INPUT_NAME].shape
            assert temp_batch.shape == s, f"Input size {temp_batch.shape} does not match model size {s}"

            self.binding_addrs[self.INPUT_NAME] = int(temp_batch.data_ptr())

            # Execute inference
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings[self.OUTPUT_NAME].data
            resultant_features.append(features.clone())

        if len(resultant_features) == 1:
            return resultant_features[0]
        else:
            rslt_features = torch.cat(resultant_features, dim=0)
            rslt_features = rslt_features[: input_tensor.shape[0]]
            return rslt_features

    def __call__(self, input_tensor):
        return self.inference(input_tensor)
    
    def eval(self, ):
        # for compatibility
        return 
 