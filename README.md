# MSSTN
This repo is the code for our paper *MSSTN: Multi-Scale Spatial-Temporal Network for Air Pollution Prediction*. We provide pre-processed data and trained models that can reproduce main result listed in our paper. Please contact us at wu-zy18@mails.tsinghua.edu.cn if you have any question.

### Requirement
We have tested our code under centos7, python3, and tensorflow 1.8.0. Similiar environment and later versions may also work but we didn't test that.

### Data
We provide pre-processed data on [Baidu NetDisk](https://pan.baidu.com/s/1cln1jpLJYP9BdBH7irrWUg) (Secure Code: 057p). Download data and replace the '/data/' folder before use. 

### Usage
##### Train
Use following command to train models from scratch:
```shell
python3 main.py train
```
##### Test
Use following command to load trained models and show result on test set:
```shell
python3 main.py test [InferenceModel]
```
where InferenceModel can be found below in config part.

##### Config
You may modify `config.yaml` to tune the training process by yourself. For example, item 'Target_City' decide which city to optimize/test when 'city_number' is set to 1, and 'InferenceModel' claim the trained model to load. More specifically, this table shows the relationship between them:

City|Index|InferenceModel
--|--|--
Beijing|0|MSSTN20190802_225240
Shijiazhuang|1|MSSTN20190803_085115
Taiyuan|2|MSSTN20190803_090722
Huhot|3|MSSTN20190803_092719
Dalian|4|MSSTN20190803_094349
