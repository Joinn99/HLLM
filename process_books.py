# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import argparse
from tqdm import tqdm
import gzip

def preprocess_interaction(intercation_path, item_path, output_path, prefix='books'):
    ratings = pd.read_csv(
        intercation_path,
        sep=",",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    item = pd.read_csv(item_path)
    ratings = ratings[ratings['item_id'].isin(item['item_id'])]
    print(f"{prefix} #data points before filter: {ratings.shape[0]}")
    print(
        f"{prefix} #user before filter: {len(set(ratings['user_id'].values))}"
    )
    print(
        f"{prefix} #item before filter: {len(set(ratings['item_id'].values))}"
    )

    # filter users and items with presence < 5
    item_id_count = (
        ratings["item_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings[ratings['timestamp'] >= '2012-12-01']
    ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
    ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
    ratings = ratings[ratings["item_count"] >= 5]
    ratings = ratings[ratings["user_count"] >= 5]
    ratings = ratings.groupby('user_id').filter(lambda x: len(x['item_id']) >= 5)
    ratings["timestamp"] = ratings["timestamp"].astype(int) // 10**9
    print(f"{prefix} #data points after filter: {ratings.shape[0]}")

    print(
        f"{prefix} #user after filter: {len(set(ratings['user_id'].values))}"
    )
    print(
        f"{prefix} #item ater filter: {len(set(ratings['item_id'].values))}"
    )
    ratings = ratings[['item_id', 'user_id', 'timestamp']]
    ratings.to_csv(output_path, index=False, header=True)


def preprocess_item(item_path, output_path, prefix='books'):
    data = []
    for line in tqdm(gzip.open(item_path, 'r')):
        json_data = eval(line)
        item_id = json_data.get('asin', '')
        description = json_data.get('description', '')
        title = json_data.get('title', '')
        if title:
            data.append({
                'item_id': item_id,
                'description': description,
                'title': title
            })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="/data")
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    args = parser.parse_args()
    file_path = args.file_path
    domain = args.domain
    domains = [
        # "Electronics",
        # "Clothing_Shoes_and_Jewelry",
        "Movies_and_TV",
        # "Cell_Phones_and_Accessories",
        # "Video_Games",
    ]
    for domain in domains:
        preprocess_item(f"{file_path}/meta_{domain}.json.gz", f"information/amazon_{domain}.csv", domain)
        preprocess_interaction(f"{file_path}/ratings_{domain}.csv", f"information/amazon_{domain}.csv", f"dataset/amazon_{domain}.csv", domain)
        df = pd.read_csv(f'dataset/amazon_{domain}.csv')
        item = pd.read_csv(f'information/amazon_{domain}.csv')
        valid_item = df['item_id'].unique()
        item = item[item['item_id'].isin(valid_item)]
        item.to_csv(f'information/amazon_{domain}.csv', index=False)