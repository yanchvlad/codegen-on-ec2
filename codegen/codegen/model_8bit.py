import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenConfig


from settings import USE_GPU, NAME_OR_PATH_TO_MODEL

torch.device(type='cuda')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print('Importing 8bit model')
print('model', NAME_OR_PATH_TO_MODEL)

class CodeModel:

    def __init__(self):
        self._config = CodeGenConfig.from_pretrained(NAME_OR_PATH_TO_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=NAME_OR_PATH_TO_MODEL,
            # config=self._config,
            device_map="auto", 
            load_in_8bit=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(NAME_OR_PATH_TO_MODEL)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        # self._model.eval()
        # self._model.to(device)

        

    def generate(self, data):
        tokenized = self._tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024,
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        # with torch.no_grad():
        model_output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length = 100
            )
        output = self._tokenizer.decode(model_output[0], skip_special_tokens=True)
        return output



        # input_ids = input_ids
        # attention_mask = attention_mask

        # with torch.no_grad():
        #     model_output = self._model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_length = 200
        #     )
        #     output = self._tokenizer.decode(model_output[0], skip_special_tokens=True)
        #     return output


code_model = CodeModel()
