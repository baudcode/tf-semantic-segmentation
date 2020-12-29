#### Convert the model

```shell
python -m tf_semantic_segmentation.bin.convert_tflite -i logs/mymodel/saved_model/0/ -o model.tflite
```

#### Test inference on the model

```shell
python -m tf_semantic_segmentation.debug.tflite_test -m model.tflite -i Harris_Sparrow_0001_116398.jpg
```