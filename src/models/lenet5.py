'''
lenet5.py

Création du modèle d'architecture LeNet-5
'''

# Initiation du modèle LeNet-5 de base
inputs = Input(shape = (28, 28, 1))

conv1 = Conv2D(filters = 30, kernel_size = (5, 5), activation = 'relu')(inputs)
pool1 = AveragePooling2D(pool_size = (2, 2))(conv1)

conv2 = Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu')(pool1)
pool2 = AveragePooling2D(pool_size = (2, 2))(conv2)

dropout = Dropout(rate = 0.2)(conv2)

flat = Flatten()(dropout)

dense1 = Dense(units = 128, activation = 'relu')(flat)
dense2 = Dense(units = 4, activation = 'softmax')(dense1)

LeNet = Model(inputs = inputs, outputs = dense2)

LeNet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
training_history_lenet = LeNet.fit(X_train, y_train, 
                                   epochs = 50, 
                                   batch_size = 256, 
                                   validation_split = 0.2)