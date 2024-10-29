import segmentation_models as sm
import prepare_dataset from dataset


BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

x_train, y_train, x_val, y_val = prepare_dataset(...)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=# TODO: simple BCE?,
    metrics=[sm.metrics.iou_score],
)

model.fit(
   x=x_train,
   y=y_train,
   batch_size=16,
   epochs=5,
   validation_data=(x_val, y_val),
)