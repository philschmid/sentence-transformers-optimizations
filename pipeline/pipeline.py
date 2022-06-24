from typing import  Dict,List, Any
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import  AutoTokenizer

def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]
class PreTrainedPipeline():
    def __init__(self, path=""):
        # load the optimized model
        self.model = ORTModelForFeatureExtraction.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=128)


    def __call__(self, inputs: Any) -> Dict[str, List[float]]:
        # tokenize the input
        encoded_input = self.tokenizer(inputs, padding="longest", truncation=True, return_tensors='pt')
        # run the model
        model_output = self.model(**encoded_input, return_dict=True)
        embeddings = cls_pooling(model_output)

        return {"vectors": embeddings[0].tolist()}
    