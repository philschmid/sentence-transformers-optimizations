from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer, ORTOptimizer
from transformers import AutoTokenizer
from pathlib import Path 
import shutil
import tempfile

# path stuff
model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
tmp_onnx = Path("onnx")

with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname=Path(tmpdirname)
    # load vanilla transformers and convert to onnx
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # save onnx checkpoint and tokenizer
    model.save_pretrained(tmpdirname)
    tokenizer.save_pretrained(tmpdirname)

    # create ORTOptimizer and define optimization configuration
    dynamic_optimizer = ORTOptimizer.from_pretrained(model_id, feature=model.pipeline_task)
    dynamic_optimization_config = OptimizationConfig(optimization_level=99)  # enable all optimizations

    # apply the optimization configuration to the model
    dynamic_optimizer.export(
        onnx_model_path=tmpdirname / "model.onnx",
        onnx_optimized_model_output_path=tmpdirname / "model-optimized.onnx",
        optimization_config=dynamic_optimization_config,
    )

# # create ORTQuantizer and define quantization configuration
# dynamic_quantizer = ORTQuantizer.from_pretrained(model_id, feature=model.pipeline_task)
# dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# # apply the quantization configuration to the model
# dynamic_quantizer.export(
#     onnx_model_path=dynamic_onnx_path / "model-optimized.onnx",
#     onnx_quantized_model_output_path=dynamic_onnx_path / "model-quantized.onnx",
#     quantization_config=dqconfig,
# )


# save model to hub
    with tempfile.TemporaryDirectory() as tmp_hub_dir:
        tmp_hub_dir=Path(tmp_hub_dir)

        repository_id = "msmarco-distilbert-base-tas-b-onnx"
        model_file_name = "model-optimized.onnx"
        model = ORTModelForFeatureExtraction.from_pretrained(tmpdirname)
        tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
        
        model.save_pretrained(tmp_hub_dir)
        tokenizer.save_pretrained(tmp_hub_dir)
        shutil.copyfile("pipeline/pipeline.py", tmp_hub_dir.joinpath("pipeline.py"))


        model.push_to_hub(tmp_hub_dir, repository_id=repository_id, use_auth_token=True)