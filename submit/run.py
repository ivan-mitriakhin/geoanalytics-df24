#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
from typing import Annotated
from warnings import simplefilter
from transform import generate_features
import joblib
import pandas as pd
import numpy as np
import typer
from typer import Option
import json

app = typer.Typer()

def main(
  # hexses_target_path: Annotated[
  #   Path, Option('--hexses-target-path', '-ht', dir_okay=False, help='Список локаций таргета', show_default=True, exists=True)
  # ] = 'hexses_target.lst',
  # hexses_data_path: Annotated[
  #   Path, Option('--hexses-data-path', '-hd', dir_okay=False, help='Список локаций транзакций', show_default=True, exists=True)
  # ] = 'hexses_data.lst',
  # input_path: Annotated[
  #   Path, Option('--input-path', '-i', dir_okay=False, help='Входные данные', show_default=True, exists=True)
  # ] = 'moscow_transaction_data01.parquet',
  # output_path: Annotated[
  #   Path, Option('--output-path', '-o', dir_okay=False, help='Выходные данные', show_default=True)
  # ] = 'output.parquet',
  ):
    with open('D:\Kaggle\Competitions\datafusion\data\hexses_target.lst', "r") as f: # open(hexses_target_path, "r") as f:
        hexses_target = [x.strip() for x in f.readlines()]
    with open('D:\Kaggle\Competitions\datafusion\data\hexses_data.lst', "r") as f: # open(hexses_data_path, "r") as f:
        hexses_data = [x.strip() for x in f.readlines()]
    hexses_suburb = pd.read_parquet('hexses_suburb.parquet')['suburb'].to_dict()
    
    transactions = pd.read_parquet('D:\Kaggle\Competitions\datafusion\data\\transactions.parquet') # pd.read_parquet(input_path)
    test_data = generate_features(transactions, hexses_data, hexses_target, hexses_suburb)
    submit = test_data[['customer_id']]
    test_data = test_data.drop('customer_id', axis=1)

    model_0_0 = joblib.load('models/model_0_0.pkl')
    model_0_1 = joblib.load('models/model_0_1.pkl')
    model_0_2 = joblib.load('models/model_0_2.pkl')

    prediction = (model_0_0.predict_proba(test_data) + model_0_1.predict_proba(test_data) + model_0_2.predict_proba(test_data)) / 3
    
    submit[hexses_target] = prediction
    

    submit.to_parquet('submit.parquet')# submit.to_parquet(output_path)

if __name__ == '__main__':
  typer.run(main, )
