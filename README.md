# FRBRT
Library of trained models for Fast Radio Burst detection

Depdencies:
```
cuda > 8.0
cudnn > 5.2
tensorflow > 1.4
blimpy > 1.1.7 (for offline inference only)
```

Sample usage:
```
python inference_multibeam.py --model=./models/molonglo.pb --filterbank_dir=/path/to/filterbank/files/
```
