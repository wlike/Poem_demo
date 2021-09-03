import os
import sys
from tqdm import tqdm
from jiuge_lvshi import Poem

genre2ch = {0: "五言绝句", 1: "七言绝句", 2: "五言律诗", 3: "七言律诗"}

if len(sys.argv) != 5:
    print('usage: {} model_path model_config vocab_file genre'.format(sys.argv[0]))
    print('    genre: {0, 1, 2, 3}, corresponding to 五言绝句、七言绝句、五言律诗、七言律诗')
    exit(-1)

model_path = sys.argv[1]
model_config = sys.argv[2]
vocab_file = sys.argv[3]
genre = int(sys.argv[4])

# 每行表示一个标题/主题
keywords = []
with open("keywords.txt", "r", encoding="utf-8") as f:
    keywords = f.readlines()
    keywords = [item.strip() for item in keywords]

# 每行表示一种前缀（如果是绝句，需要4个汉字；如果是律诗，需要8个汉字）
prefixes = []
if os.access("prefixes.txt", os.R_OK):
    with open("prefixes.txt", "r", encoding="utf-8") as f:
        prefixes = f.readlines()
        prefixes = [item.strip() for item in prefixes]

poem = Poem(model_path, model_config, vocab_file)

results = []
if len(prefixes) == 0:
    for item in tqdm(keywords):
        result = poem.generate(title=item, prefix=None, genre=genre)
        print("result: {}".format(result))
        results.append(item + " # " + result)
else:
    for item in tqdm(keywords):
        for pre in tqdm(prefixes):
            result = poem.generate(title=item, prefix=pre, genre=genre)
            print("result: {}".format(result))
            results.append(item + ' # ' + result)

res_file = "result_" + genre2ch[genre] + ".txt"
with open(res_file, 'w', encoding='utf-8') as f:
    for res in results:
        f.write(res + "\n")