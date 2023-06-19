import os

def transform_seq(seq):
    ids = list()
    cnt = 0
    for ch in seq:
        if ch == '-':
            ids.append(-1)
        else:
            ids.append(cnt)
            cnt += 1
    return ids

def compute_f1(pred_seq, golden_seq):
    pred_seq1, pred_seq2 = pred_seq
    golden_seq1, golden_seq2 = golden_seq
    assert len(pred_seq1) == len(pred_seq2)
    assert len(golden_seq1) == len(golden_seq2)
    pred_seq1_ids = transform_seq(pred_seq1)
    pred_seq2_ids = transform_seq(pred_seq2)
    pred_align_ids = set([(a, b) for a, b in zip(pred_seq1_ids, pred_seq2_ids)])
    golden_seq1_ids = transform_seq(golden_seq1)
    golden_seq2_ids = transform_seq(golden_seq2)
    golden_align_ids = set([(a, b) for a, b in zip(golden_seq1_ids, golden_seq2_ids)])
    correct = len(pred_align_ids & golden_align_ids)
    golden_all = len(golden_align_ids)
    pred_all = len(pred_align_ids)
    return correct, pred_all, golden_all
    # return f1, precision, recall
    
def get_pred_seq(file_path):
    pred_seq_list = list()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            seq1, seq2 = line.split('\t')
            pred_seq_list.append((seq1, seq2))
    return pred_seq_list

def get_golden_seq(file_path):
    golden_seq_list = list()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            _, seq, label = line.split('\t')
            label_list = label.split(',')
            label_list = list(map(int, label_list))
            align_seq = ""
            for i, ch in enumerate(seq):
                align_seq = align_seq + '-' * label_list[i] + ch
            align_seq += '-' * label_list[-1]
            golden_seq_list.append(align_seq)
    return golden_seq_list

if __name__ == "__main__":
    golden_seq1_list = get_golden_seq("data/test_0.txt")
    golden_seq2_list = get_golden_seq("data/test_1.txt")
    golden_seq_list = [(a, b) for a, b in zip(golden_seq1_list, golden_seq2_list)]
    
    pred_seq_list = get_pred_seq("align.txt")
    # result_f = open("cross-attention-f1.txt", 'w')
    correct_list = list()
    pred_all_list = list()
    golden_all_list = list()
    for pred_seq, golden_seq in zip(pred_seq_list, golden_seq_list):
        correct, pred_all, golden_all = compute_f1(pred_seq, golden_seq)
        correct_list.append(correct)
        pred_all_list.append(pred_all)
        golden_all_list.append(golden_all)
    correct_num = sum(correct_list)
    pred_all_num = sum(pred_all_list)
    golden_all_num = sum(golden_all_list)
    precision = 1.0 * correct_num / pred_all_num
    recall = 1.0 * correct_num / golden_all_num
    f1 = 2 * precision * recall / (precision + recall)
    # result_f.write("f1: %.3f\tprecision: %.3f\trecall: %.3f\n"%(f1, precision, recall))
    print("f1: %.3f\tprecision: %.3f\trecall: %.3f\n"%(f1, precision, recall))
    # result_f.close()
