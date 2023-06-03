import re
import torch

from transformers import T5Tokenizer

class MSADataset(torch.utils.data.Dataset):
    def __init__(self, msa_file_list, tokenizer, msa_num=2, max_length=512, data_type="train"):
        self.msa_file_list = msa_file_list
        self.tokenizer = tokenizer
        self.msa_num = msa_num
        assert len(msa_file_list) == msa_num
        if msa_num != 2:
            raise NotImplementedError("msa_nmu > 2 not supported.")
        self.prot1_list = list()
        self.prot2_list = list()
        self.label1_list = list()
        self.label2_list = list()

        seq_num = 0
        skip_num = 0
        with open(msa_file_list[0]) as f1, open(msa_file_list[1]) as f2:
            for item1, item2 in zip(f1, f2):
                seq_num += 1
                item1_list = item1.split("\t")
                prot1, label1 = self.do_tokenize(item1_list[1], item1_list[2])
                item2_list = item2.split("\t")
                prot2, label2 = self.do_tokenize(item2_list[1], item2_list[2])
                if len(prot1) > max_length or len(prot2) > max_length:
                    prot1 = prot1[:max_length]
                    prot2 = prot2[:max_length]
                    label1 = label1[:max_length]
                    label2 = label2[:max_length]
                    skip_num += 1
                self.prot1_list.append(prot1)
                self.label1_list.append(label1)
                self.prot2_list.append(prot2)
                self.label2_list.append(label2)
        print("[%s] %d found, %d skipped(%.5f)." % (data_type, seq_num, skip_num, 1.0 * skip_num / seq_num))

    def do_tokenize(self, seq, label):
        seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        seq = self.tokenizer(seq, return_tensors="pt")["input_ids"][0]
        label = torch.tensor(list(map(int, label.split(","))), dtype=torch.float)
        assert seq.shape[0] == label.shape[0] # 追加了</s>, tokens数量应当是相同的
        return seq, label

    def __len__(self):
        return len(self.prot1_list)

    def __getitem__(self, idx):
        seq1 = self.prot1_list[idx]
        label1 = self.label1_list[idx]
        seq2 = self.prot2_list[idx]
        label2 = self.label2_list[idx]
        return idx, seq1, seq2, label1, label2

    def _collate_tensors(self, tensors):
        max_len = max(tensor.size(0) for tensor in tensors)
        # B, T, padding=0
        out = tensors[0].new_zeros((len(tensors), max_len))
        for i, v in enumerate(tensors):
            out[i, : v.size(0)] = v
        return out

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _, _ in samples], dtype=torch.long)
        seq1s = self._collate_tensors([seq for _, seq, _, _, _ in samples])
        length1s = torch.tensor([len(seq) for _, seq, _, _, _ in samples], dtype=torch.long)
        seq2s = self._collate_tensors([seq for _, _, seq, _, _ in samples])
        length2s = torch.tensor([len(seq) for _, _, seq, _, _ in samples], dtype=torch.long)
        label1s = self._collate_tensors([label for _, _, _, label, _ in samples])
        label2s = self._collate_tensors([label for _, _, _, _, label in samples])
        out = {
            "id": indices,
            "seq1": seq1s,
            "seq2": seq2s,
            "label1": label1s,
            "label2": label2s,
            "length1": length1s,
            "length2": length2s
        }
        return out

def build_data_iter(msa_file_list, data_type="train", batch_size=32):
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    dataset = MSADataset(msa_file_list, tokenizer, data_type=data_type)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader

if __name__ == '__main__':
    msa_file_list = ["data/train_0.txt", "data/train_1.txt"]
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    # ret = tokenizer.batch_encode_plus(["A A A A", "A A A A A A A A A A A"], add_special_tokens=True, padding="longest")
    # print(ret)
    dataset = MSADataset(msa_file_list, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_size=2,
        shuffle=True
    )
    # for d in train_loader:
    #     print(d)
    #     exit(0)
