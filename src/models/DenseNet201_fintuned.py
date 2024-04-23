'''
DenseNet201_fintuned.py

Ce script permet de construire le modèle DenseNet201 avec ses paramétrages fins.
'''

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Construction du modèle
densenet = DenseNet201(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

# Dégel de couches pour l'ajustement fin
for layer in densenet.layers[-32:]:
    layer.trainable = True

# Ajout de la tête de classification
x = densenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation = 'relu', kernel_regularizer = l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs = densenet.input, outputs = predictions)

model.compile(
    optimizer = Adam(learning_rate = 1e-4),
    loss = categorical_crossentropy,
    metrics = [Precision(), Recall(), AUC(), F1Score()]
)

early_stopper = EarlyStopping(
    monitor = 'val_loss',    # Surveiller la perte de validation
    min_delta = 0.001,       # Seuil pour juger qu'une amélioration est significative
    patience = 5,            # Nombre d'époques sans amélioration après lesquelles l'entraînement sera arrêté
    verbose = 1,             # Afficher les messages d'arrêt
    mode = 'auto',
    restore_best_weights = True 
)

# Entraînement du modèle avec les générateurs
history = model.fit(
    x = X_train,
    y = y_train,
    validation_data = (X_test, y_test),
    batch_size = 32,
    shuffle = True,
    epochs = 50,
    verbose = 1,
    callbacks = [early_stopper]
)