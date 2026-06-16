# infer 
```
## input s1 and s2, mix them, then seprate the mix
python infer_two_sources.py --s1 asserts/audios/gt/1221_3575/spk1_reverb.wav --s2 asserts/audios/gt/1221_3575/spk2_reverb.wav  --conf_dir configs/spmamba-echo2mix.yml

```

```
## input s1 and mix, s1 is target, then seprate the mix
python infer_two_sources.py --mix /mnt/e/data/AVSEC/avsec3/scenes/S50000_mixed.wav --s1 /mnt/e/data/AVSEC/avsec3/scenes/S50000_target.wav --output_dir /mnt/e/data/AVSEC/avsec3/spmamba/S5000
```