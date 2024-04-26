import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import layers

def build_model_VGG16(hp):

    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False
    
    #test profondeur
    for layer in base_model.layers[-8:]:
        layer.trainable = True


    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(hp.Choice('units', [128,256, 512, 1024, 2056]), activation='relu')(x)
    x = Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1))(x) #ajout dropout
    predictions = layers.Dense(4, activation='softmax')(x)  

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5, 1e-6])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model