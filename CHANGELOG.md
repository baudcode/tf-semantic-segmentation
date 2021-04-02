Release 0.3.2
===============

Major Features
- Line detection
- Mlflow integration
- New Models: Deeplab variants

Models
- DeeplabV3
- DeeplabV3PlusXception
- DeeplabV3PlusMobile

Augmentations:
- rot90

Features
- line detection on binary classification & logging to tensorboard
- update mixed precision api
- mlflow integration
- automatic logdir naming based on parameters
- line detection extra requirements
- Add SequenceDataset wrapper 


Release 0.3.1
===============

Fixes

- prediction scaling issue in tensorboard
- updated colab link

Release 0.3.0
===============

Major Features
- Documentation
- New experimental models: Unet++, PSP, FCN, U2Net
- Tensorflow 2.4 support


Models
- Unet++
- PSP
- FCN
- U2Net


Datasets
- BioimageBenchmark

Fixes
- Renamed Plastic ds to Taco ds
- Fixed train.py errors
- Fixed mixed precision training

Features
- Add learning rate finder
- Add optimal batch size finder
- Improved dataset loading
- Add patch as resize method
- Add colab notebook and 

Release 0.2.3
===============

Models:
- SatelliteUnet
- ImagenetUnets (with pretrained weights)
- UnetV2 (removed)

Major Changes
- model fn returns only 1 model

Features
- Fixes MultiresUnet (no bn at the end)
- Use logger instead of print
- ReduceLROnPlateau
- TFLite Exporter
- Extended serving (writing config, ensemble)
- Allow using dataset and model in the trainer

Release 0.2.2
===============

Models:
- MultiresUnet

Datasets
- Isic2018
- cub2002011
- shapesdsmini
- cvc_clinicdb

Activations
- LeakyReLU

Features
- Added tests (50% coverage)
- Use TensorFlow summary methods instead of Tensorboardx
- Fixes visualization errors
- Show predictions as a stream of images
- Predict on video


Release 0.2.1
===============
- Fix pypi automatic release
- Allow preprocessing without mask

Release 0.2.0
===============

- Added Cityscapes
- Create tfrecord from directory
- Upgrade tensorflow to version 2.1
- Predictions logging to tensorboard

Release 0.1.0
===============

Initial Release