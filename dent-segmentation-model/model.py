import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import keras
import segmentation_models as sm
from dataset import prepare_datasets
import matplotlib.pyplot as plt


def main():
    '''
    Compile and build model
    '''
    
    BACKBONE = 'vgg19'
    EPOCHS = 5
    BATCH_SIZE = 16

    print('Preparing training and validation datasets...')
    train_dataset_loader, val_dataset_loader = prepare_datasets(BATCH_SIZE)
        
    x_batch, y_batch = train_dataset_loader[0]
    print(f"Training points shape: {x_batch.shape}, Train masks shape: {y_batch.shape}")

    x_val_batch, y_val_batch = val_dataset_loader[0]
    print(f"Validation points shape: {x_val_batch.shape}, Validation masks shape: {y_val_batch.shape}")

    print('Compiling model...')
    model = sm.Unet(BACKBONE, encoder_weights=None, input_shape=(None, None, 2))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss='dice',
        metrics=[sm.metrics.iou_score]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.weights.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau()
    ]

    print('Building model...')
    history = model.fit(
        train_dataset_loader,
        # steps_per_epoch=len(train_dataset_loader),
        epochs=EPOCHS,
        callbacks = callbacks,
        validation_data=val_dataset_loader,
        # validation_steps=len(val_dataset_loader)
    )

    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
