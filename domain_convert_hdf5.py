import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import PyTorchModelHubMixin
from transformers import logging
import subprocess
import json
import os
import glob
import h5py
import numpy as np

from accelerate import Accelerator
from accelerate.utils import gather_object

# 设置模型和设备
MODEL_NAME = "nvidia/domain-classifier"  # 替换为你需要的模型
# DATA_FILE = "EleutherAI/the_pile_deduplicated"
# TARGET_MODEL = "TinyLlama-TinyLlama_v1.1"
# print("The target model is {}".format(TARGET_MODEL))
# hdf5_tokenizer_path = "/home/pretraining/klyang/mount_dir/mount/Llama-3-8B"
hdf5_tokenizer_path = "/mnt/blob-hptrainingwesteurope-pretraining/Llama-3-8B"

# DATA_FILE = "/mnt/klyang_data/quality_classification/{}/temperature-1.0".format(TARGET_MODEL)
# DATA_FILE = "/home/pretraining/klyang/mount_dir/mount/dolmino-mix-1124/math_split_8"
# DATA_FILE = "/mnt/blob-hptrainingwesteurope-pretraining/dolmino-mix-1124/math_split_8"
DATA_FILE = "/mnt/blob-hptrainingwesteurope-pretraining/fineweb-edu-RAND-100B-split-64"
source_split_num = 64
target_split_num = 8

assert source_split_num % target_split_num == 0

max_sample_num_per_file = 30000
# DATA_FILE = "/home/pretraining/klyang/mount_dir/mount/quality_classification/{}/temperature-1.0".format(TARGET_MODEL)

# SAVE_DIR = "/mnt/klyang_data/filtered_generated_data/{}/temperature-1.0".format(TARGET_MODEL)  # 替换为你需要的模型
# SAVE_DIR = "/mnt/blob-hptrainingwesteurope-pretraining-out/dolmino-mix-1124/math_domain_split_8"  # 替换为你需要的模型
SAVE_DIR = "/mnt/blob-hptrainingwesteurope-pretraining-out/fineweb-edu-RAND-100B-domain-split-8"  # 替换为你需要的模型
# SAVE_DIR = "/home/pretraining/klyang/mount_dir/mount/filtered_generated_data/{}/temperature-1.0".format(TARGET_MODEL)  # 替换为你需要的模型

# SAVE_DIR = "/home/pretraining/klyang/pretrain_data_mixing/domain_classification/pile/labels_{}.jsonl".format(nu)  # 替换为你需要的模型
# DISTRIBUTION_SAVE_FILE = "/home/pretraining/klyang/pretrain_data_mixing/domain_classification/distribution.jsonl"
# TARGET_TOKEN_NUM = 1000000
# PER_GPU_BATCH_SIZE = 64  # 每批推理的样本数
PER_GPU_BATCH_SIZE = 32  # 每批推理的样本数

def write_to_hdf5(split, file_path, data_dict, rank):
    """
    Append a list of dictionaries to a JSONL file.

    Parameters:
        file_path (str): Path to the JSONL file.
        dict_list (list): List of dictionaries to append.

    Returns:
        None
    """
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be dictionaries.")

    # Open the file in append mode and write each dictionary as a new line
    # with open(os.path.join(file_path, "labels_{}.jsonl".format(nu)), 'w+', encoding='utf-8') as file:
    for item in data_dict.keys():
        if len(data_dict[item]) == 0:
                continue
        save_path = os.path.join(file_path, "split_{}".format(split), item)
        os.makedirs(save_path, exist_ok=True)
        count = 0
        for k in range(0, len(data_dict[item]), max_sample_num_per_file):
            cur_data = data_dict[item][k:min(k+max_sample_num_per_file, len(data_dict[item]))]
            with h5py.File(os.path.join(save_path, "rank_{}_{}_{}.hdf5".format(rank, count, len(cur_data))), "w") as f:
                array = np.stack(data_dict[item], axis=0)
                f.create_dataset('train', data=array)
            count += 1
        

def load_hdf5_data(filename, tokenizer):
    data_list = []
    if filename.endswith(".hdf5"):
        print(filename)
        file_path = os.path.join(dir_name, filename)
        with h5py.File(file_path, 'r') as hdf5_file:
            for item in hdf5_file['train']:
                decoded_text = tokenizer.decode(item)
                data_list.append({'text': decoded_text, 'id': item})
    return data_list
                    

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    accelerator = Accelerator()
    num_processes = accelerator.num_processes

    # 配置日志
    logging.set_verbosity_info()
    logger = logging.get_logger()
    
    logger.info("加载 tokenizer...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hdf5_tokenizer = AutoTokenizer.from_pretrained(hdf5_tokenizer_path)
    
    logger.info("加载模型并分配设备...")
    model = CustomModel.from_pretrained(MODEL_NAME)
    model.eval()
    
    # devices = get_available_devices()
    # logger.info(f"可用设备: {devices}")

    model.to("cuda:{}".format(accelerator.process_index))
    print('current process: {}'.format(accelerator.process_index))

    batch_size = PER_GPU_BATCH_SIZE * num_processes

    domain_split_dict = {}
    for item in config.label2id.keys():
        domain_split_dict[item] = []

    for i in range(source_split_num):
        if accelerator.is_main_process:
            print("Processing for {}-th split.".format(i))
        
        dir_name = os.path.join(DATA_FILE, "split_{}".format(str(i).zfill(2)))
        hdf5_files = [os.path.join(root, file) for root, _, files in os.walk(dir_name) for file in files]

        for file_name in hdf5_files:
            data = load_hdf5_data(file_name, hdf5_tokenizer)

            for j in range(0, len(data), batch_size):
                batch = data[j:min(j+batch_size, len(data))]

                if accelerator.is_main_process:
                    print("{}/{} instances processed.".format(j+1, len(data)))

                with accelerator.split_between_processes(batch) as samples:
                    # print("num_samples: {}".format(len(samples)))
                    ids = [sample['id'] for sample in samples]
                    texts = [sample['text'] for sample in samples]
                    inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True).to("cuda:{}".format(accelerator.process_index))

                outputs = model(inputs["input_ids"], inputs["attention_mask"])

                predicted_classes = torch.argmax(outputs, dim=1)
                predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()]
                assert len(ids) == len(predicted_domains)

                for s_id, predicted_domain in zip(ids, predicted_domains):
                    domain_split_dict[predicted_domain].append(s_id)
        
        if (i+1) % int(source_split_num/target_split_num) == 0:
            t_split_num = int((i+1)/int(source_split_num/target_split_num))-1
            print("Saving! source split {}, target split {}".format(i, t_split_num))

            write_to_hdf5(t_split_num, SAVE_DIR, domain_split_dict, accelerator.process_index)
            domain_split_dict = {}
            for item in config.label2id.keys():
                domain_split_dict[item] = []

    print("All data saved for rank {}".format(accelerator.process_index))

if __name__ == "__main__":
    main()