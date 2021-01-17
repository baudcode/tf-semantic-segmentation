### Losses

- Catagorical Crossentropy
- Binary Crossentropy
- Crossentropy + SSIM
- Dice
- Crossentropy + Dice
- Tversky
- Focal
- Focal + Tversky

##### Code

```
from tf_semantic_segmentation.losses import get_loss_by_name, losses_by_name

losses = list(losses_by_name.keys())
for l in losses:
    fn = get_loss_by_name(l)
    value = fn(y_true, y_pred)
```

### Metrics

- f1
- f2
- iou
- precision
- recall
- psnr
- ssim

#### Using Code

```
from tf_semantic_segmentation.metrics import get_metric_by_name, metrics_by_name

metrics = list(metrics_by_name.keys())
for m in metrics:
    fn = get_metric_by_name(m)
    value = fn(y_true, y_pred)
```