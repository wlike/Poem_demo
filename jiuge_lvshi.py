import json
import torch

from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
from generator import CheckedGenerator
from poemcheck import Checker
from constants import *

class Poem():
    def __init__(self, model_path=None, model_config=None, vocab_file='cache/vocab_with_title.txt'):
        self.model = None
        if model_path is not None and model_config is not None:
            self.load_model(model_path, model_config)
        else:
            self.load_model()
        self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        self.checker = Checker()
        with open('cache/label_to_id.json', 'r', encoding='utf-8') as f:
            self.title_to_ids = json.load(f)
        print('title_to_ids: {}'.format(self.title_to_ids))

    def load_model(self, model_path='cache/model/model_epoch_1.pt',
                   model_config='cache/model_config.json', device='cpu'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('device: {}'.format(self.device))

        model_config = GPT2Config.from_json_file(model_config)
        model = GPT2LMHeadModel(config=model_config)
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def generate(self, title='无题', prefix=None, genre=1):
        if self.model is None:
            raise Exception("has no model")

        temperature = 1
        topk = 0

        context_tokens = []
        assert genre in [0, 1, 2, 3]

        text_genre_list = ['五言绝句', '七言绝句', '五言律诗', '七言律诗']
        genre_code_list = ['wuyanjue', 'qiyanjue', 'wuyanlv', 'qiyanlv']
        length = [4*5, 4*7, 8*5, 8*7]

        text_genre = text_genre_list[genre]
        genre_code = genre_code_list[genre]

        ids = self.title_to_ids[text_genre]
        context_tokens.append(ids)
        print('context_tokens: {}'.format(context_tokens))

        context_tokens.append(MASK_IDX)

        print('title: {}'.format(title))

        context_tokens.extend(
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(title)))
        context_tokens.append(JING_IDX)
        print('context_tokens: {}'.format(context_tokens))

        if prefix is not None:
            prefix = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prefix))

        out = None
        while out is None:
            generator = CheckedGenerator(model=self.model,
                                         context=context_tokens,
                                         prefix=prefix,
                                         tokenizer=self.tokenizer,
                                         checker=self.checker,
                                         genre=genre_code,
                                         temperature=temperature,
                                         top_k=topk, device=self.device)
            out = generator.sample_sequence_v2(length[genre])
        out = out.tolist()
        print('out: {}'.format(out))

        text = self.tokenizer.convert_ids_to_tokens(out)
        text = ''.join(text)
        return text

if __name__ == '__main__':
    lvshi = Poem()
    print(lvshi.generate())
