The dataset is a zip file split into 10 chunks, and must first be combined into a single zip archive before extraction. 

After download all data from GD:

```shell
for file in *.zip; do unzip "$file"; done
```

To fully unpack the dataset:

```shell
cd DDR_dataset
cat DDR-dataset.zip.0* > DDR-dataset.zip
unzip DDR-dataset.zip
```