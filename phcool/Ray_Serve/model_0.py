# File name: serve_quickstart.py
from starlette.requests import Request


from transformers import pipeline
import time


class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small",device="cuda")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

translator=Translator()
begin_time=time.time()

for i in range(100):
    output=translator.translate("hello world!")
    print(i)


end_time=time.time()
duration=end_time-begin_time
print(duration)
