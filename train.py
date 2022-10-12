import argparse
import json
import os.path

# from accelerate import Accelerator
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdafactorSchedule

from datasets import get_dataset_functions
from src.unifiedqa_trainer import UnifiedQATrainer
from src.utils import prepare_run_files_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train T5 unifiedQA model')
    parser.add_argument('--experiment',
                        type=str,
                        default='check',
                        help='Experiment to run from experiments/config/exp_<experiment.json')
    parser.add_argument('--evaluate-date',
                        default=False,
                        action='store_true', dest='evaluate_date',
                        help='run evaluation on model predictions as date')
    parser.add_argument('--no-answer-dataset',
                        default=False,
                        action='store_true',
                        dest='no_answer_dataset',
                        help='whether the training dataset has <no answer> samples or not')
    parser.add_argument('--save-dir',
                        type=str,
                        default='./run_files',
                        action='store',
                        dest='save_dir')
    args = parser.parse_args()

    with open(os.path.join('experiments/configs', f'exp_{args.experiment}.json')) as f:
        configs = json.load(f)

    model_size = configs['model']['size']

    no_answer_dataset = args.no_answer_dataset

    run_files = args.save_dir
    prepare_run_files_directory(run_files)

    logfile = os.path.join(f'experiments/logs/exp_{args.experiment}.tsv')

    # accelerator = Accelerator()
    # device = accelerator.device
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if n_gpu > 0 else 'cpu')
    print(f'Running on device: {device}')
    if n_gpu > 0:
        print(f'Number of GPUs: {n_gpu}')

    model_name = f'allenai/unifiedqa-v2-t5-{model_size}-1251000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        torch.cuda.manual_seed_all(42)
    if n_gpu > 0:
        model.to(torch.device("cuda"))
    if n_gpu == 0:
        model.to(torch.device("cpu"))

    train_datasets = {}
    for dataset in configs['train_datasets']:
        dataset_type, dataset_name = dataset.split(':')
        train_datasets[dataset] = get_dataset_functions[dataset_type](dataset_name)

    train_dataloader = DataLoader(ConcatDataset(train_datasets.values()), shuffle=True,
                                  batch_size=configs['train_batch_size'])

    eval_datasets = {}
    for dataset in configs['eval_datasets']:
        dataset_type, dataset_name = dataset.split(':')
        eval_datasets[dataset] = get_dataset_functions[dataset_type](dataset_name)

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    trainer = UnifiedQATrainer(run_files,
                               model,
                               tokenizer,
                               optimizer,
                               lr_scheduler,
                               device,
                               eval_batch_size=configs['eval_batch_size'],
                               no_answer_dataset=no_answer_dataset,
                               n_gpu=n_gpu)
    config_string = f'Run configs from experiments/configs/exp_{args.experiment}.json: {configs}'

    with open(logfile, 'w') as f:
        f.write(f'Dataset\tEpoch\tPrecision\tRecall\tF1\tAccuracy\n')

    print('-' * 50)
    print(config_string)
    print('*' * 50)
    for dataset_name, dataset in list(train_datasets.items()) + list(eval_datasets.items()):
        pos_count, neg_count = dataset.summary()
        print(f'Dataset: {dataset_name} pos_count: {pos_count} neg_count: {neg_count}')
    print('*' * 50)

    print(f'Pre fine-tuning evaluations:')
    for dataset_name, dataset in eval_datasets.items():
        trainer.evaluate(-1, dataset_name, dataset, logfile, evaluate_dates=args.evaluate_date)
        print('-' * 10)

    for epoch in range(configs['epochs']):
        print('-' * 50)
        trainer.train(train_dataloader, epoch)
        # trainer.train(train_datasets['train'], epoch)
        print('-' * 10)
        for dataset_name, dataset in eval_datasets.items():
            trainer.evaluate(epoch, dataset_name, dataset, logfile, evaluate_dates=args.evaluate_date)
            print('-' * 10)
