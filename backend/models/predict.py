 
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


 
# # Load Data

 
from preprocess_data import get_data_generators  # Ensure this file is in the same directory
train_gen, val_gen, test_gen = get_data_generators()

 
# # Define model

 
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False  # Freeze base model

for layer in base_model.layers[-20:]:  # ✅ Unfreeze the last 20 layers
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


 
# # Compile model

 
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

 
# # Callbacks

 
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')


 
# # train model

 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate Model

eval_results = model.evaluate(test_gen)
print(f"Test Loss: {eval_results[0]:.4f}, Test Accuracy: {eval_results[1]:.4f}")


 
# Save Final Model
 
model.save("final_skin_model.keras")
print("Model training complete and saved!")


