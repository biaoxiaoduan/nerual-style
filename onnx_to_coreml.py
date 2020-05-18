import sys
from onnx import onnx_pb
from onnx_coreml import convert

model_in = sys.argv[1]
model_out = sys.argv[2]

model_file = open(model_in, 'rb')
content = model_file.read()
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(content)
coreml_model = convert(model_proto, image_input_names=['input.1'], image_output_names=['178'], minimum_ios_deployment_target='13')
coreml_model.save(model_out)
