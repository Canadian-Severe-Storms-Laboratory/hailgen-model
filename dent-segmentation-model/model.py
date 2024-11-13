import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
from dataset import prepare_dataset


BACKBONE = 'vgg19'

print('Preparing dataset...')
x_train, y_train, x_val, y_val = prepare_dataset()

print('Compiling model...')
model = sm.Unet(BACKBONE, encoder_weights=None, input_shape=(None, None, 2))
model.compile(
    'Adam',
    loss='binary_crossentropy',
    metrics=[sm.metrics.iou_score]
)

print('Building model...')
model.fit(
   x=x_train,
   y=y_train,
   batch_size=16,
   epochs=1,
   validation_data=(x_val, y_val)
)

model.save('dent-segmentation-model.keras')
