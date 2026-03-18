print("scrpit started")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "../dataset",
    target_size=(224,224),
    batch_size=32,
    subset="training"
)

val = datagen.flow_from_directory(
    "../dataset",
    target_size=(224,224),
    batch_size=32,
    subset="validation"
)

print("Classes:", train.class_indices)
print("Train samples:", train.samples)
print("Validation samples:", val.samples)