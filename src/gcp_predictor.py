import os
import pickle
import logging
from typing import List

import torch
import tiktoken
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

from model import GPT
from generate import generate_batch


class GPTPredictor(Predictor):
    def __init__(self):
        self.logger = logging.getLogger()
    
    def load(self, artifacts_uri: str):
        prediction_utils.download_model_artifacts(artifacts_uri) # downloads all files from local or bucket dir to cwd
        
        self.logger.debug(f'DEBUG artifact_uri: {artifacts_uri}')
        self.logger.debug(str(os.listdir('./')))
        
        with open('./tiktoken_config.pkl', 'rb') as f:
             tiktoken_config = pickle.load(f)
        self.enc = tiktoken.core.Encoding(
                tiktoken_config['name'],
                explicit_n_vocab=tiktoken_config['explicit_n_vocab'],
                pat_str=tiktoken_config['pat_str'],
                mergeable_ranks=tiktoken_config['mergeable_ranks'],
                special_tokens=tiktoken_config['special_tokens'],
            )

        with open('./model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        self.logger.debug(f'Device: {device}')

        self.model = GPT(
            model_config['vocab_dim'],
            model_config['sequence_dim'],
            model_config['embed_dim'],
            model_config['num_heads'],
            model_config['num_layers'],
            dropout=model_config['dropout'],
            device=device,
        )
        self.model.load('./gpt.pth', map_location=device)

    def preprocess(self, prediction_input: dict) -> List[str]:
        instances = prediction_input["instances"] # instances must be List[str] to conform to google API
        return instances

    @torch.inference_mode()
    def predict(self, instances: List[str]) -> torch.Tensor:
        prediction_results = generate_batch(self.model, self.enc.encode, self.enc.decode, instances, 1000, print_batch_num=0)
        return prediction_results

    def postprocess(self, prediction_results: torch.Tensor) -> dict:
        results = []
        for i in range(prediction_results.shape[0]):
            results.append(self.enc.decode(prediction_results[i].tolist()))
        return {"predictions": results}