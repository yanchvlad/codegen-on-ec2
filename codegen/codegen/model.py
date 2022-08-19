import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PretrainedConfig


from settings import USE_GPU, NAME_OR_PATH_TO_MODEL

torch.device(type='cuda')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('DEVICE')
print(device)

class CodeModel:
    _model: AutoModelForSeq2SeqLM
    _tokenizer: AutoTokenizer

    def __init__(self):
        # self._config = GPT2Config.from_pretrained(NAME_OR_PATH_TO_MODEL)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=NAME_OR_PATH_TO_MODEL,
            # config=self._config,
            device_map="auto", 
            load_in_8bit=True
        ).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(NAME_OR_PATH_TO_MODEL)

    # self._model.eval()
    # if USE_GPU:
    #     self._model.to('cuda')

    def generate(self, data):
        tokenized = self._tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024,
        ).to(device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']


        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            model_output = self._model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length = 200
            ).to_cuda(device)
            output = self._tokenizer.decode(model_output[0], skip_special_tokens=True)
            return output


code_model = CodeModel()
