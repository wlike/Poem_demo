import torch
import torch.nn.functional as F
import time
import numpy as np
from constants import *
from utils import *

class BaseGenerator(object):
    def __init__(self, model, context, prefix, tokenizer, temperature=1, top_k=0, device='cpu'):
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.repetition_penalty = 1.2

        self.generated = self.get_context_tensor(context)
        self.context_len = len(self.generated[0])
        print("context_len: {}".format(self.context_len))

    def filtering(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1
        # 将已生成的字设置为 -Inf，以避免重复生成
        generated = self.generated[0].tolist()
        generated = [index for index in generated if index != PAD_IDX and index != COMMA_IDX]
        for index in generated:
            logits[index] /= self.repetition_penalty
        # 限制可生成字的范围大小
        if self.top_k > 0:
            # 等价于: logits < torch.topk(logits, self.top_k)[0][-1].unsqueeze(0)
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        return logits

    def sample_sequence(self):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated}
                outputs = self.model(**inputs) 
                next_token_logits = outputs[0][0, -1, :] / self.temperature
                filtered_logits = self.filtering(next_token_logits)
                while True:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token.tolist() != [UNK_IDX]:
                        break
                if next_token.tolist() == [PAD_IDX]:
                    break
                self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
        return self.generated

    def get_context_tensor(self, context):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        return context

class CheckedGenerator(BaseGenerator):
    def __init__(self, model, context, prefix, tokenizer, checker, genre, temperature=1, top_k=0, device='cpu'):
        super(CheckedGenerator, self).__init__(model, context, prefix, tokenizer, temperature, top_k, device)
        self.checker = checker
        self.pattern_label = None
        self.pattern = None
        self.genre = genre
        self.subgenre = 'lv' if len(genre) == 7 else 'jue'
        self.genre_to_length = {"wuyanlv": 5, "qiyanlv": 7, "wuyanjue": 5, "qiyanjue": 7}
        self.count = -1
        self.position = 0
        self.yun = None

        # 获取非中文字符的ID
        non_ch_ids = []
        for i in range(self.tokenizer.vocab_size):
            term = self.tokenizer.convert_ids_to_tokens(i)
            if not is_chinese(term):
                non_ch_ids.append(i)
        self.non_ch_ids = torch.tensor(non_ch_ids, dtype=torch.long, device=self.device)

        print("prefix: {}".format(prefix))
        self.prefix = prefix
        self.prefix_idx = 0
        # 当前生成的总汉字个数
        self.ch_cnt = 0
        # 每句的目标字数
        self.target_interval_length = self.genre_to_length[self.genre]
        # 当前单句生成的汉字个数
        self.cur_interval_length = 0

        if self.prefix is not None:
            if self.subgenre == 'lv':
                assert len(self.prefix) == 8
            else:  # self.subgenre == 'jue'
                assert len(self.prefix) == 4

    def filtering_with_check(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1
        tokens = self.tokenizer.convert_ids_to_tokens(self.generated[0].tolist())
        if tokens[-1] == '，':
            self.position = 0
        else:
            self.position += 1
        # 首句生成完成
        if self.pattern_label == None and tokens[-1] == '，':
            self.pattern_label = self.checker.judge_pattern(tokens[(-1-self.target_interval_length):-1], self.subgenre)
            if self.pattern_label == None:
                print("首句未匹配，重新生成")
                return None
            print('pattern_label: {}'.format(self.pattern_label))
            pingze = self.pattern_label[-1]
            print("pingze: {}".format(pingze))
            # 首句入韵
            if pingze == '1':
                self.yun = tokens[-2]
                print("odd_yun: {}".format(self.yun))
            self.pattern = self.checker.getpattern(self.pattern_label, self.subgenre)
            return self.filtering_with_labels(logits)
        else:
            # 首句不入韵时，根据第二句确定韵
            # 绝句第二、四句押韵，首句(第一句)可押可不押
            # 律诗第二、四、六、八句押韵，首句(第一句)可押可不押
            if tokens[-1] == '，' and self.yun == None:
                self.yun = tokens[-2]
                print("even_yun: {}".format(self.yun))

            if self.pattern_label == None:
                return super().filtering(logits)
            else:
                return self.filtering_with_labels(logits)

    def filtering_with_labels(self, logits):
        self.count += 1
        if self.count >= len(self.pattern):
            return super().filtering(logits)
        else:
            current = self.pattern[self.count]
            if current == ' ':
                self.count += 1
                current = self.pattern[self.count]
            if current == '0':
                return super().filtering(logits)
            elif current == '1':
                if self.position == self.target_interval_length - 1 and self.yun != None:
                    logits = self.filteringYun(logits)
                return self.filteringPing(logits)
            else:  # current == '2'
                return self.filteringZe(logits)

    def filteringPing(self, logits, filter_value=-float('Inf')):
        pingindex = self.checker.get_zeindex(self.tokenizer)
        logits[pingindex] = filter_value
        return super().filtering(logits)

    def filteringZe(self, logits, filter_value=-float('Inf')):
        zeindex = self.checker.get_pingindex(self.tokenizer)
        logits[zeindex] = filter_value
        return super().filtering(logits)

    def filteringYun(self, logits, filter_value=-float('Inf')):
        entire = set(range(len(logits)))
        yun = self.checker.get_yunindex(self.yun, self.tokenizer)
        left = entire - set(yun)
        logits[list(left)] = filter_value
        return logits

    def sample_sequence(self):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated}
                outputs = self.model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / self.temperature
                filtered_logits = self.filtering_with_check(next_token_logits)
                if filtered_logits is None:
                    return None
                while True:
                    logits = np.array(F.softmax(filtered_logits, dim=-1).tolist())
                    logits = logits[logits > 0]
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token.tolist() != [UNK_IDX]:
                        break
                print('next_token: {}'.format(next_token))
                if next_token.tolist() == [PAD_IDX]:
                    break
                self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
                print('generated: {}'.format(self.generated))
        return self.generated

    def sample_sequence_v2(self, length):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated[0].unsqueeze(0)}
                outputs = self.model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / self.temperature
                next_token_logits[torch.isnan(next_token_logits)] = -float('Inf')
                # 将非中文字符的 logits 设置为最小，进而不会生成
                next_token_logits[self.non_ch_ids] = -float('Inf')
                # 将非前缀字的 logits 设置为最小，进而可以生成前缀字
                if self.prefix is not None and self.ch_cnt % self.target_interval_length == 0:
                    print("ch_cnt: {}".format(self.ch_cnt))
                    print("target_interval_length: {}".format(self.target_interval_length))
                    print("prefix_idx: {}".format(self.prefix_idx))
                    print("prefix_token_idx: {}".format(self.prefix[self.prefix_idx]))
                    print("prefix_orig_value: {}".format(next_token_logits[self.prefix[self.prefix_idx]]))
                    token_ids = torch.tensor([i for i in range(self.tokenizer.vocab_size)], dtype=torch.long, device=self.device)
                    non_prefix_ids = (token_ids != self.prefix[self.prefix_idx])
                    next_token_logits[non_prefix_ids] = -float('Inf')
                    self.prefix_idx += 1
                filtered_logits = self.filtering_with_check(next_token_logits)
                if filtered_logits is None:
                    return None
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                print('next_token: {}'.format(next_token))
                next_token_str = self.tokenizer.convert_ids_to_tokens(next_token)[0]
                print("next_token_str: {}".format(next_token_str))
                if self.cur_interval_length < self.target_interval_length:
                    if is_chinese(next_token_str):
                        self.ch_cnt += 1
                        self.cur_interval_length += 1
                        self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
                # 单句生成结束
                if self.cur_interval_length == self.target_interval_length:
                    # 添加单句结束标志“，”，为了 poemcheck.py
                    comma = torch.tensor(self.tokenizer.convert_tokens_to_ids("，"), dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
                    self.generated = torch.cat((self.generated, comma), dim=1)
                    # 准备下一句（可能需要添加前缀）
                    self.cur_interval_length = 0
                if self.ch_cnt == length:
                    break
        return self.generated[0][self.context_len : ]
