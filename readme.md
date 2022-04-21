# ASRL: An Adaptive GPS Sampling Method Using Deep Reinforcement Learning

## Paper

This is a TensorFlow implementation of the ASRL Adaptive GPS Sampling Method. For more details, please refer to our paper **ASRL: An Adaptive GPS Sampling Method Using Deep Reinforcement Learning**, which has been accepted at The 23rd IEEE International Conference on Mobile Data Management. If this code is useful for your work, please cite our paper:

```
@inproceedings{
    ASRL2022,
    title={ASRL: An Adaptive GPS Sampling Method Using Deep Reinforcement Learning},
    author={Boting, Qu and Mengjiao, Zhao and Jun, Feng and Xin, Wang},
    booktitle={IEEE International Conference on Mobile Data Management},
    year={2022},
}
```

## Requirements

- tensorflow >=2.0.0
- tensorflow-probability 0.6.0
- tensorlayer >=2.0.0
- pandas
- gym

## Data Preparation

##### Data for Model Training

There are three datasets (i.e., [Seattle Dataset](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/), [San Francisco Dataset](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) and [Global Dataset](https://ieeexplore.ieee.org/document/7225829)) used in this experiment, all of them are available online. Before training ASRL, the GPS points should be preprocessed first. Here is an example:

| longitude | latitude | time | speed |
| :-------: | :------: | :---: | :---: |
|   54.96   |  56.25  | 12:01 |   0   |
|   54.93   |  56.25  | 12:02 |   3   |
|   54.98   |  56.25  | 12:03 |   0   |
|    ...    |   ...   |  ...  |  ...  |

Please note that since there is no GPS speed value in the Global dataset, it needs to be calculated first before the model training.

To represent the object moving status, the trajectory GPS points and road vertices are encoded using the Geohash algorithm. GPS points and road vertices with the same geohash code are considered to be in the same grid. Then the positioning error and road density of each grid are calculated. Here is an example using Geohash length of 7. Please refer to `resource/` folder for all the positioning error and road density data of the Global dataset.

| geohash | positioning_error |
| :-----: | :---------------: |
| v4p048q | 5.574366022296077 |
|   ...   |        ...        |

| geohash | road_density |
| :-----: | :----------: |
| v1yx49k |      10      |
|   ...   |     ...     |

##### Data for IRL

To learn the reward function by IRL, the Expert Trajectories and Random Trajectories are needed. Here we provide these trajectories of the Global dataset in `/IRL/resourse/` folder. We also provide `state_detail.txt` which specifies the road density, positioning error, moving orientation change and speed to represent the state. Here is an example:

| road_density | positioning_error | time | speed |
| :----------: | :---------------: | :--: | :---: |
|      1      |         1         |  0  |   2   |
|      1      |         1         |  3  |   2   |
|      2      |         1         |  3  |   2   |
|     ...     |        ...        | ... |  ...  |

## Run the Pre-trained Model

The pre-trained model are stored in `model/` folder.

Before running the model, the environment that agent interacts with should be build using the gym package. For more details, please refer to `https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym` . You may need `/enviroment/myenv.py` .

After building the environment, change `data_path` in `myenv.py` into your own GPS points storage path and run `model/program_test.py` directly.

## Model Training

After building the environment, change `data_path` in `myenv.py` into your own GPS points storage path and `run_log` in `program_start.py` as experiment log. Then run `model/program_start.py` directly.

## **Acknowledgements**

Thanks to `MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/` for getting me started with the code for the DDPG.
