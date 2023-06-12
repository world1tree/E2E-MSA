import torch
from utils.logging import init_logger, logger
from torch.optim import Adam
from build_dataset import build_data_iter
from build_model import build_model
from dataclasses import dataclass, field

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    return n_params

@dataclass
class ModelConfig(object):
    src_word_vec_size:int=field(
        default=512,
        metadata={
            "help": "Word embedding size for src."
        }
    )
    tgt_word_vec_size:int=field(
        default=512,
        metadata={
            "help": "Word embedding size for tgt."
        }
    )
    share_embeddings:bool=field(
        default=True,
        metadata={
            "help": "Share the word embeddings between encoder "
                    "and decoder. Need to use shared dictionary for this "
                    "option."
        }
    )
    position_encoding:bool=field(
        default=True,
        metadata={
            "help": "Use a sin to mark relative words positions."
                    "Necessary for non-RNN style models."
        }
    )
    enc_layers:int=field(
        default=6,
        metadata={
            "help": "Number of layers in the encoder"
        }
    )
    dec_layers:int=field(
        default=6,
        metadata={
            "help": "Number of layers in the decoder"
        }
    )
    enc_rnn_size:int=field(
        default=512,
        metadata={
            "help": "Size of encoder rnn hidden states."
                    "Must be equal to dec_rnn_size except for"
                    "speech-to-text."
        }
    )
    dec_rnn_size:int=field(
        default=512,
        metadata={
            "help": "Size of decoder rnn hidden states."
                    "Must be equal to dec_rnn_size except for"
                    "speech-to-text."
        }
    )
    self_attn_type:str=field(
        default="scaled-dot",
        metadata={
            "help": "Self attention type in Transformer decoder"
                    "layer -- currently 'scaled-dot' or 'average'"
        }
    )

    heads:int=field(
        default=8,
        metadata={
            "help": "Number of heads for transformer self-attention"
        }
    )

    transformer_ff:int=field(
        default=2048,
        metadata={
            "help": "Size of hidden transformer feed-forward"
        }
    )

    dropout:float=field(
        default=0.1,
        metadata={
            "help": "Dropout probability; applied in LSTM stacks."
        }
    )

def model_forward(model, criterion, seq1, seq2, seq1_length, label, g_label, padding_idx):
    # seq1经过encoder0, seq2经过encoder1, 预测seq2的结果
    output1 = model(seq1, seq2, seq1_length)[0]
    pred = model.generator(output1)
    # 非padding为1, padding为0
    pred_mask = (~(seq2.eq(padding_idx))).to(torch.float)
    # 排除掉padding的预测结果
    pred = pred * pred_mask.unsqueeze(-1)
    ntokens = torch.sum(pred_mask).item()
    # 计算平均loss
    loss = criterion(label, pred[:,:,0]) / ntokens
    g_loss = criterion(g_label, pred[:,:,1]) / ntokens

    # 计算预测准确率
    label_round = torch.round(label)
    pred_round = torch.round(pred[:,:,0])
    correct_ntokens = torch.sum((label_round == pred_round) * pred_mask, dtype=torch.long).item()

    return loss, g_loss, correct_ntokens, ntokens, pred_round, pred_mask

def do_valid(model, criterion, val_dataloader, device, padding_idx):
    logger.info("Epoch End. Running valid.")
    model.eval()
    total_dev_loss = 0.
    total_correct1 = 0
    total_correct2 = 0
    total_ntokens1 = 0
    total_ntokens2 = 0
    with torch.no_grad():
        for val_data in val_dataloader:
            seq1 = val_data["seq1"].to(device)
            seq2 = val_data["seq2"].to(device)
            length1 = val_data["length1"].to(device)
            length2 = val_data["length2"].to(device)
            label1 = val_data["label1"].to(device)
            label2 = val_data["label2"].to(device)
            g_label1 = val_data["g_label1"].to(device)
            g_label2 = val_data["g_label2"].to(device)
            # 预测seq2对应的label
            avg_loss1, avg_g_loss1, correct1, ntokens1, _, _ = model_forward(model, criterion, seq1, seq2, length1, label2, g_label2, padding_idx)
            # 预测seq1对应的label
            avg_loss2, avg_g_loss2, correct2, ntokens2, _, _ = model_forward(model, criterion, seq2, seq1, length2, label1, g_label1, padding_idx)
            dev_loss = avg_loss1 + avg_loss2 + avg_g_loss1 + avg_g_loss2
            total_dev_loss += dev_loss.item()
            total_correct1 += correct1
            total_correct2 += correct2
            total_ntokens1 += ntokens1
            total_ntokens2 += ntokens2
        logger.info(f'''Dev Loss: {total_dev_loss / len(val_dataloader): .3f} Dev Acc1: {1.0 * total_correct1 / total_ntokens1: .3f} Dev Acc2: {1.0 * total_correct2 / total_ntokens2: .3f}''')
    model.train()

def do_test(model, criterion, test_dataloader, device, padding_idx, result_file="align.txt"):

    def _get_alignments(seq, length, pred):
        seq = seq.cpu()
        length = length.cpu()
        pred = pred.cpu()
        align_list = list()
        for i, pred_single in enumerate(pred):
            align_str = ""
            p = pred_single[:length[i]]
            seq_ids_without_padding = seq[i][:length[i]-1].tolist()
            tokens = model.encoder.embeddings.tokenizer.convert_ids_to_tokens(seq_ids_without_padding)
            tokens = list(map(lambda ch: ch.lstrip('▁'), tokens))
            for j, token in enumerate(tokens):
                freq = int(p[j].item())
                align_str = align_str + "-" * freq + token
            freq = int(p[-1].item()) # last
            align_str += "-" * freq
            align_list.append(align_str)
        return align_list

    logger.info("Train End. Running test.")
    model.eval()
    total_correct1 = 0
    total_correct2 = 0
    total_ntokens1 = 0
    total_ntokens2 = 0
    align_list1 = list()
    align_list2 = list()
    with torch.no_grad():
        for test_data in test_dataloader:
            seq1 = test_data["seq1"].to(device)
            seq2 = test_data["seq2"].to(device)
            length1 = test_data["length1"].to(device)
            length2 = test_data["length2"].to(device)
            label1 = test_data["label1"].to(device)
            label2 = test_data["label2"].to(device)
            g_label1 = test_data["g_label1"].to(device)
            g_label2 = test_data["g_label2"].to(device)
            # 预测seq2对应的label
            _, _, correct1, ntokens1, pred1, pred_mask1 = model_forward(model, criterion, seq1, seq2, length1, label2, g_label2, padding_idx)
            # 预测seq1对应的label
            _, _, correct2, ntokens2, pred2, pred_mask2 = model_forward(model, criterion, seq2, seq1, length2, label1, g_label1, padding_idx)
            total_correct1 += correct1
            total_correct2 += correct2
            total_ntokens1 += ntokens1
            total_ntokens2 += ntokens2

            # 序列1的对齐信息
            _align_list1 = _get_alignments(seq1, length1, pred2)
            # 序列2的对齐信息
            _align_list2 = _get_alignments(seq2, length2, pred1)
            align_list1.extend(_align_list1)
            align_list2.extend(_align_list2)
        f = open(result_file, "a")
        for s1, s2 in zip(align_list1, align_list2):
            s1 = s1.lstrip("-")
            s2 = s2.lstrip("-")
            max_len = max(len(s1), len(s2))
            if len(s1) < max_len:
                s1 = s1 + "-" * (max_len-len(s1))
            if len(s2) < max_len:
                s2 = s2 + "-" * (max_len-len(s2))
            result_s1 = ""
            result_s2 = ""
            for a, b in zip(s1, s2):
                if a == "-" and b == "-":
                    continue
                result_s1 += a
                result_s2 += b
            f.write(result_s1 + "\t" + result_s2 + "\n")
        f.close()

        logger.info(f'''Test Acc1: {1.0 * total_correct1 / total_ntokens1: .3f} Test Acc2: {1.0 * total_correct2 / total_ntokens2: .3f}''')
    model.train()

def train(model, train_dataloader, val_dataloader, test_dataloader, learning_rate, epochs, log_steps):
    # 混合精度设置
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    padding_idx = model.encoder.padding_idx
    # 通过Dataset类获取训练和验证集
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.998))
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 日志输出
    train_epoch_loss = 0.
    total_correct1 = 0
    total_correct2 = 0
    total_ntokens1 = 0
    total_ntokens2 = 0
    train_step_loss = 0.
    step_correct1 = 0
    step_correct2 = 0
    step_ntokens1 = 0
    step_ntokens2 = 0
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 进度条函数tqdm
        for step, train_data in enumerate(train_dataloader, start=1):
            seq1 = train_data["seq1"].to(device)
            seq2 = train_data["seq2"].to(device)
            length1 = train_data["length1"].to(device)
            length2 = train_data["length2"].to(device)
            label1 = train_data["label1"].to(device)
            label2 = train_data["label2"].to(device)
            g_label1 = train_data["g_label1"].to(device)
            g_label2 = train_data["g_label2"].to(device)

            with autocast():
                # 预测seq2对应的label
                avg_loss1, avg_g_loss1, correct1, ntokens1, _, _ = model_forward(model, criterion, seq1, seq2, length1, label2, g_label2, padding_idx)
                # 预测seq1对应的label
                avg_loss2, avg_g_loss2, correct2, ntokens2, _, _ = model_forward(model, criterion, seq2, seq1, length2, label1, g_label1, padding_idx)
            # 记录用于日志输出
            total_correct1 += correct1
            total_correct2 += correct2
            total_ntokens1 += ntokens1
            total_ntokens2 += ntokens2
            step_correct1 += correct1
            step_correct2 += correct2
            step_ntokens1 += ntokens1
            step_ntokens2 += ntokens2

            batch_loss = avg_loss1 + avg_loss2 + avg_g_loss1 + avg_g_loss2
            # 计算损失
            train_epoch_loss += batch_loss.item()
            train_step_loss += batch_loss.item()
            # 计算精度
            # 模型更新
            model.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # batch_loss.backward()
            # optimizer.step()

            # 输出训练信息
            if step % log_steps == 0:
                logger.info(f'''Epochs: {epoch_num + 1} Step: {step} Train Loss: {train_step_loss / log_steps: .3f} Train Acc1: {1.0 * step_correct1 / step_ntokens1: .3f} Train Acc2: {1.0 * step_correct2 / step_ntokens2: .3f}''')
                train_step_loss = 0.
                step_correct1 = 0
                step_correct2 = 0
                step_ntokens1 = 0
                step_ntokens2 = 0
            # logger.info(f'''Epochs: {epoch_num + 1} Step: {step} Train Loss: {train_step_loss / log_steps: .3f} Train Acc1: {1.0 * step_correct1 / step_ntokens1: .3f} Train Acc2: {1.0 * step_correct2 / step_ntokens2: .3f}''')
        train_epoch_loss = 0.
        total_correct1 = 0
        total_correct2 = 0
        total_ntokens1 = 0
        total_ntokens2 = 0

        # # ------ 验证模型 -----------
        do_valid(model, criterion, val_dataloader, device, padding_idx)
    do_test(model, criterion, test_dataloader, device, padding_idx)

if __name__ == '__main__':
    EPOCHS = 5
    LOG_STEPS = 5
    # 初始化logger
    init_logger()
    model = build_model(ModelConfig())
    nparams = _tally_parameters(model)
    logger.info("number of parameters: %d" % nparams)
    LR = 1e-4
    train_dataloader = build_data_iter(["data/train_0.txt", "data/train_1.txt"], data_type="train", batch_size=32)
    valid_dataloader = build_data_iter(["data/valid_0.txt", "data/valid_1.txt"], data_type="valid", batch_size=32)
    test_dataloader = build_data_iter(["data/test_0.txt", "data/test_1.txt"], data_type="test", batch_size=32, shuffle=False)
    train(model, train_dataloader, valid_dataloader, test_dataloader, LR, EPOCHS, LOG_STEPS)
