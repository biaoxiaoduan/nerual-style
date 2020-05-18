python3 ./neural_style/neural_style.py eval --content-image images/content-images/stata.jpg --output-image tmp.jpg --model $1 --cuda 0 --export_onnx tmp.onnx
python3 ./onnx_to_coreml.py tmp.onnx $2.mlmodel
#rm tmp.onnx
