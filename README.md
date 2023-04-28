# PAMRec: Playback duration Augmented Micro-video Recommendation

This is the official implementation of our RecSys'23 paper:  

The code is tested under a Linux desktop with TensorFlow 2.4.0 and Python 3.8.6.

## Model Training

Use the following command to train a PAMRec model on `WeChat-Channels` dataset: 

```
python examples/00_quick_start/sequential.py --dataset wechat
```

or on `MX-TakaTak` dataset:

```
python examples/00_quick_start/sequential.py --dataset takatak
``` 

## Note

The implemention is based on *[Microsoft Recommender](https://github.com/microsoft/recommenders)*.
