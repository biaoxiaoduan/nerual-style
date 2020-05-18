import sys
import coremltools
from coremltools.models.neural_network.quantization_utils import *
import os
def quantize(fin, bits, functions, fout):
    model = coremltools.models.MLModel(fin)
    for function in functions :
        for bit in bits:
            sys.stdout.flush()
            quantized_model = quantize_weights(model, bit, function)
            sys.stdout.flush()
            #quantized_model.author = "Alexis Creuzot"
            #quantized_model.short_description = str(bit)+"-bit per quantized weight, using "+function+"."
            #quantized_model.save(fout)
            coremltools.utils.save_spec(quantized_model, fout)

def qtz(fin, fout):
    model_spec = coremltools.utils.load_spec(fin)
    model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
    coremltools.utils.save_spec(model_fp16_spec, fout)


flist = os.listdir('.')
for f in flist:
    if f.endswith('mlmodel'):
        fin = f
        fout = 'qtz/%s' % f
        quantize(fin, [8], ['linear'], fout)
        #qtz(fin, fout)
