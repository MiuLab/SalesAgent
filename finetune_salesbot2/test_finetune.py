import os
import tqdm
import torch
import wandb
import argparse
from utils.datasets import SalesBot2Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
def main(args):
    # Create dataset
    #init gpt2
    # model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = SalesBot2Dataset(args.data_path, tokenizer, args.max_length)
    # Create dataloader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=dataset.collate_fn)
    # # create optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # print(dataset.__len__())
    # create wandb
    # wandb.init(project="salesbot2")
    # wandb.watch(model)
    # # train
    # print(model.transformer.wte.weight.shape[0] == len(tokenizer))
    # for epoch in range(10):
    #     for batch in tqdm.tqdm(dataloader):
    #         print(batch['input_ids'].shape)
    #         print(batch['target_ids'].shape)
    #         optimizer.zero_grad()
    #         input_ids = batch['input_ids'].to(args.device)
    #         labels = batch['target_ids'].to(args.device)
    #         outputs = model(input_ids, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         wandb.log({'loss': loss.item()})
    #     # save models
    #     model.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)






def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument('--data_path', type=str, default='../MSGD_dataset_final_sanity_1024.json')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--num_beams', type=int, default=5)

    args = parser.parse_args()
    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    return args


if __name__ == "__main__":
    args = parse_arg()
    main(args)
