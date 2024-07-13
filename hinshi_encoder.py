import MeCab
from transformers import AutoTokenizer, LlamaTokenizer
import random


def build_hinshi_tokenize(tokenizer, rate=0.5, add_special_tokens=True):
    HINSHI = [
        "感動詞",
        "記号",
        "形容詞",
        "助詞",
        "助動詞",
        "接続詞",
        "動詞",
        "副詞",
        "名詞",
        "連体詞"
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': [f"<{h}>" for h in HINSHI]})
    mecabTagger = MeCab.Tagger("-Ochasen")

    def get_hinshi(char):
        node = mecabTagger.parseToNode(char)
        while node:
            hinshi = node.feature.split(",")[0]
            if hinshi in HINSHI:
                return hinshi
            node = node.next
    
    def encode(text):
        tokenized = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text)
        ids = []
        for char,id in zip(tokenized, encoded):
            rand = random.randint(0, 100)
            if rate*100 > rand:
                h = get_hinshi(char)
                h_id = tokenizer.encode(f'<{h}>')[0]
                ids.append(h_id)
            else:
                ids.append(id)
        if add_special_tokens:
            ids += [tokenizer.eos_token_id]
        return ids

    return encode

def main():
    text='形態素解析したい文章を入力します'
    # tokenizer = LlamaTokenizer.from_pretrained("NovelAI/nerdstash-tokenizer-v2")
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v2.0")
    
    # tokenizer.add_tokens([f"<{h}>" for h in hinshi], special_tokens=True)

    # text_tokenized = tokenizer.encode(text, add_special_tokens=False)
    # print(text_tokenized)
    # print(tokenizer.decode(text_tokenized))

    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)
    # print(tokenizer('a<形容詞>'))

    # text_tokenized = tokenizer.tokenize(text)
    # print(text_tokenized)
    encode = build_hinshi_tokenize(tokenizer, rate=0.2)
    r = encode(text)
    print(r)
    print(tokenizer.decode(r))


if __name__ == '__main__':
    main()
    