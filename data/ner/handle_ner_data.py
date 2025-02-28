import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                trans_label = {}
                for key, items in label.items():
                    items = items.keys()
                    trans_label[key] = list(items)
                trans = {
                    "text": text,
                    "label": trans_label
                }
                line = json.dumps(trans, ensure_ascii=False)
                w.write(line + "\n")
                w.flush()

trans("data/ner/train_origin.json", "data/ner/train.json")
trans("data/ner/dev_origin.json", "data/ner/dev.json")



