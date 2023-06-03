import torch
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

def model_forward(model, criterion, seq1, seq2, seq1_length, label, padding_idx):
    # seq1经过encoder0, seq2经过encoder1, 预测seq2的结果
    output1 = model(seq1, seq2, seq1_length)[0]
    pred = model.generator(output1).squeeze(-1)
    # 非padding为1, padding为0
    pred_mask = (~(seq2.eq(padding_idx))).to(torch.float)
    # 排除掉padding的预测结果
    pred = pred * pred_mask
    ntokens = torch.sum(pred_mask).item()
    # 计算平均loss
    loss = criterion(label, pred) / ntokens

    # 计算预测准确率
    label_round = torch.round(label)
    pred_round = torch.round(pred)
    correct_ntokens = torch.sum((label_round == pred_round) * pred_mask, dtype=torch.long).item()

    return loss, correct_ntokens, ntokens


def train(model, train_dataloader, val_dataloader, learning_rate, epochs, log_steps):
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

            # 预测seq2对应的label
            avg_loss1, correct1, ntokens1 = model_forward(model, criterion, seq1, seq2, length1, label2, padding_idx)
            # 预测seq1对应的label
            avg_loss2, correct2, ntokens2 = model_forward(model, criterion, seq2, seq1, length2, label1, padding_idx)
            # 记录用于日志输出
            total_correct1 += correct1
            total_correct2 += correct2
            total_ntokens1 += ntokens1
            total_ntokens2 += ntokens2
            step_correct1 += correct1
            step_correct2 += correct2
            step_ntokens1 += ntokens1
            step_ntokens2 += ntokens2

            batch_loss = avg_loss1 + avg_loss2
            # 计算损失
            train_epoch_loss += batch_loss.item()
            train_step_loss += batch_loss.item()
            # 计算精度
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 输出训练信息
            if step % log_steps == 0:
                print(
                    f'''Epochs: {epoch_num + 1} 
                      | Step: {step}
                      | Train Loss: {train_step_loss / log_steps: .3f} 
                      | Train Accuracy1: {1.0 * step_correct1 / step_ntokens1: .3f} 
                      | Train Accuracy2: {1.0 * step_correct2 / step_ntokens2: .3f}''')
                train_step_loss = 0.
                step_correct1 = 0
                step_correct2 = 0
                step_ntokens1 = 0
                step_ntokens2 = 0
        print(
            f'''Epochs: {epoch_num + 1} 
              | Step: {step}
              | Train Loss: {train_epoch_loss / len(train_dataloader): .3f} 
              | Train Accuracy1: {1.0 * total_correct1 / total_ntokens1: .3f} 
              | Train Accuracy2: {1.0 * total_correct2 / total_ntokens2: .3f}''')
        train_epoch_loss = 0.
        total_correct1 = 0
        total_correct2 = 0
        total_ntokens1 = 0
        total_ntokens2 = 0

        # # ------ 验证模型 -----------
        # # 定义两个变量，用于存储验证集的准确率和损失
        # total_acc_val = 0
        # total_loss_val = 0
        # # 不需要计算梯度
        # with torch.no_grad():
        #     # 循环获取数据集，并用训练好的模型进行验证
        #     for val_input, val_label in val_dataloader:
        #         # 如果有GPU，则使用GPU，接下来的操作同训练
        #         val_label = val_label.to(device)
        #         mask = val_input['attention_mask'].to(device)
        #         input_id = val_input['input_ids'].squeeze(1).to(device)
        #
        #         output = model(input_id, mask)
        #
        #         batch_loss = criterion(output, val_label)
        #         total_loss_val += batch_loss.item()
        #
        #         acc = (output.argmax(dim=1) == val_label).sum().item()
        #         total_acc_val += acc

EPOCHS = 5
LOG_STEPS = 5
model = build_model(ModelConfig())
nparams = _tally_parameters(model)
print("number of parameters: %d" % nparams)
LR = 1e-4
train_dataloader = build_data_iter(["data/train_0.txt", "data/train_1.txt"], data_type="train", batch_size=16)
valid_dataloader = build_data_iter(["data/valid_0.txt", "data/valid_1.txt"], data_type="valid", batch_size=16)
train(model, train_dataloader, valid_dataloader, LR, EPOCHS, LOG_STEPS)
