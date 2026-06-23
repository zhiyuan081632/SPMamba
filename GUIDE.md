
# 生成数据
```
cd DataPreProcess

## create libri2mix datasets(min, 8k/16k)
./create_librimix_min.sh


## create echo2mix datasets
./create_echo2mic.sh


## create lrs3 datasets(with reverb)
./create_lrs3.sh --reverb
```

# 模型训练
```
python audio_train.py --conf_dir=configs/spmamba-lrs3.yml
```


# 性能评估
```
## eval for echo2mix testsets
python audio_test.py --conf_dir configs/spmamba-echo2mix.yml --output_dir output/echo2mix

## eval for libri2mix testsets
python audio_test.py --conf_dir configs/spmamba-librimix.yml --output_dir output/libri2mix
```


# 推理测试
```
## input s1 and s2, mix them, then seprate the mix
python infer_two_sources.py --s1 asserts/audios/gt/1221_3575/spk1_reverb.wav --s2 asserts/audios/gt/1221_3575/spk2_reverb.wav  --conf_dir configs/spmamba-echo2mix.yml
```

```
## input s1 and mix, s1 is target, then seprate the mix
python infer_two_sources.py --mix /mnt/e/data/AVSEC/avsec3/scenes/S50000_mixed.wav --s1 /mnt/e/data/AVSEC/avsec3/scenes/S50000_target.wav --output_dir /mnt/e/data/AVSEC/avsec3/spmamba/S5000
```

```
## input mix, then seprate the mix
python infer_two_sources.py --mix asserts/audios/gt/1221_3575_mix.wav
```