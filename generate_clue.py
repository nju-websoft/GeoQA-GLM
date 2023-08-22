# coding=utf-8
# from transformers import get_linear_schedule_with_warmup, AutoTokenizer
# from transformers import T5ForConditionalGeneration

from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import trange
import os
import random
from utils import save_dataset, set_seed, save_model, read_dataset, compute_rouges
import json
import argparse
import ast
import torch

device = torch.device("cuda:0")

def get_input_feature_train(features, tokenizer, max_gen_length):
    inputs, targets = [], []
    for sample in features:
        inputs.append(sample['question'] + ' [MASK]')
        answer = sample['answer']
        answer_idx = ord(answer) - ord('A')
        correct_opt = sample['options'][answer_idx]
        targets.append(correct_opt)
    inputs = tokenizer(
        inputs,
        return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=targets, max_gen_length=max_gen_length, padding=False)
    inputs = inputs.to('cuda')
    return inputs


def get_input_feature_test(features, tokenizer, max_gen_length):
    inputs = []
    for sample in features:
        inputs.append(sample['scenario'] + " " + sample['question'] + ' [MASK]')
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=True)
    # print(inputs)
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=max_gen_length)
    inputs = inputs.to('cuda')
    return inputs

@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, tokenizer, max_gen_len):
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    generations, glods = [], []
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index: end_index]]
        inputs = get_input_feature_test(batch_example, tokenizer, max_gen_len)
        outputs = model.generate(**inputs, max_length=max_gen_len, eos_token_id=tokenizer.eop_token_id,
                                 pad_token_id=tokenizer.eop_token_id)
        for output, sample in zip(outputs, batch_example):
            clue = tokenizer.decode(output.tolist(), skip_special_tokens=True)
            question = sample['question']
            clue = clue.replace(question, '')
            sample['clue'] = clue
            generations.append(clue)
            glods.append(sample['options'][ord(sample['answer']) - ord('A')])
    rouge_scores = compute_rouges(generations, glods)
    return rouge_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='THUDM/glm-large-chinese/',
                        type=str)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--gpu",
                        default="1",
                        type=str)
    parser.add_argument("--dataset_name",
                        default='GKMC',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--results_save_path",
                        default='results',
                        type=str)
    parser.add_argument("--output_dir",
                        default='outputs',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--init",
                        default=None,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_gen_len",
                        default=60,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    only_eval = args.only_eval
    debug = args.debug
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_path_base = f'./dataset/{dataset_name}/'

    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/dev.json'
    data_path_test = f'{data_path_base}/test.json'

    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]


    config_name = f'{args.dataset_name}/{model_name_abb}'

    parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}'
    output_model_path = f'./{args.output_dir}/{config_name}/{parameter_name}/'
    path_save_result = f'./dataset/{dataset_name}/'
    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)
    if debug:
        train_examples = read_dataset(data_path_train)[:10]
        dev_examples = read_dataset(data_path_dev)[:10]
        test_examples = read_dataset(data_path_test)[:10]
    else:
        train_examples = read_dataset(data_path_train)
        dev_examples = read_dataset(data_path_dev)
        test_examples = read_dataset(data_path_test)

    train_batch_size = args.train_batch_size

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    # model = model.half().cuda()
    model = model.cuda()
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      "epoch": args.epoch_num,
                      "train_path": data_path_train,
                      "dev_path": data_path_dev,
                      "test_path": data_path_test,
                      "train_size": len(train_examples),
                      "train_examples": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'max_gen_len': args.max_gen_len,
                      'output_model_path': output_model_path,
                      'path_save_result': path_save_result,
                      'init_checkpoint': args.init_checkpoint}, indent=2))
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    if only_eval:
        args.init = True
    if args.init and args.init_checkpoint is None:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)
    elif args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)

    if only_eval:
        scores = evaluate(model, dev_examples, args.eval_batch_size, tokenizer)
        print('dev:', scores)
        save_dataset(path_save_result, '/dev.json', dev_examples)
        
        
        scores = evaluate(model, test_examples, args.eval_batch_size, tokenizer)
        print('test:', scores)
        save_dataset(path_save_result, '/test.json', test_examples)
        exit(0)

    warm_up_ratio = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size)
    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    best_test_acc = 0
    best_dev_acc = 0
    best_dev_result, best_test_result = None, None
    if args.init_checkpoint is not None:
        scores_dev, results_dev, readable_results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer, args.max_gen_len)
        scores = sum([scores_dev[key] for key in scores_dev.keys()])
        print('scores_dev:', scores_dev)
        best_dev_acc = scores

    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        model.train()
        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            inputs = get_input_feature_train(batch_example, tokenizer, args.max_gen_len)
            outputs = model(**inputs)
            loss = outputs.loss
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4))
            step_trange.set_postfix_str(loss_show)

        scores_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer, args.max_gen_len)
        print('dev:', scores_dev)
        scores = sum([scores_dev[key] for key in scores_dev.keys()])
        if scores > best_dev_acc:
            best_dev_acc = scores
            print('save new best')
            save_model(output_model_path, model, optimizer)
            save_dataset(path_save_result, '/dev.json', dev_examples)
            scores_test = evaluate(model, test_examples, args.eval_batch_size,tokenizer, args.max_gen_len)
            best_test_result = scores_test
            best_dev_result = scores_dev
            print('test:', scores_test)
            save_dataset(path_save_result, '/test.json', test_examples)

            scores_train = evaluate(model, train_examples, args.eval_batch_size, tokenizer, args.max_gen_len)
            print('train:', scores_train)
            save_dataset(path_save_result, '/train.json', train_examples)


    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

