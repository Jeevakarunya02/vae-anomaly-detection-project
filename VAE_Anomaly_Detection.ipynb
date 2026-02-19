{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a6bc2b53-a8fd-4829-a766-c3e4316323b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\JEEVAKARUNYA T\\.anaconda\\Anaconda\\envs\\vae_env\\lib\\site-packages\\keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e612dba1-00ec-47e4-85a6-a8172378d34e",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs=15\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6f13a225-b827-4ea9-a8bf-c68b4491db8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (2.15.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.15.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow) (2.15.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.4.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=23.5.26 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (25.12.19)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.7.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=2.9.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.15.1)\n",
      "Requirement already satisfied: libclang>=13.0.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: ml-dtypes~=0.2.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.26.4)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.4.0)\n",
      "Requirement already satisfied: packaging in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (25.0)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.25.8)\n",
      "Requirement already satisfied: setuptools in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (80.10.2)\n",
      "Requirement already satisfied: six>=1.12.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.17.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.15.0)\n",
      "Requirement already satisfied: wrapt<1.15,>=1.11.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.14.2)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.78.0)\n",
      "Requirement already satisfied: tensorboard<2.16,>=2.15 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.2)\n",
      "Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)\n",
      "Requirement already satisfied: keras<2.16,>=2.15.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)\n",
      "Requirement already satisfied: google-auth<3,>=1.6.3 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.48.0)\n",
      "Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.2.4)\n",
      "Requirement already satisfied: markdown>=2.6.8 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.10.2)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.32.5)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.1.5)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.4.2)\n",
      "Requirement already satisfied: cryptography>=38.0.3 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (46.0.5)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (4.9.1)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.0.0)\n",
      "Requirement already satisfied: charset_normalizer<4,>=2 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.4.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.11)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.6.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2026.1.4)\n",
      "Requirement already satisfied: pyasn1>=0.1.3 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from rsa<5,>=3.1.4->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.6.2)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.15.0->tensorflow) (0.46.3)\n",
      "Requirement already satisfied: cffi>=2.0.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from cryptography>=38.0.3->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.0.0)\n",
      "Requirement already satisfied: pycparser in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from cffi>=2.0.0->cryptography>=38.0.3->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.0)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.3.1)\n",
      "Requirement already satisfied: markupsafe>=2.1.1 in .\\.anaconda\\anaconda\\envs\\vae_env\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.0.3)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install tensorflow\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "bba34e9a-6952-42e4-a622-e393b5d21908",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Normal Shape: (54051, 784)\n",
      "Test Combined Shape: (10000, 784)\n"
     ]
    }
   ],
   "source": [
    "# Load MNIST dataset\n",
    "(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()\n",
    "\n",
    "# Normalize pixel values (0 to 1)\n",
    "x_train = x_train.astype(\"float32\") / 255.\n",
    "x_test = x_test.astype(\"float32\") / 255.\n",
    "\n",
    "# Flatten images (28x28 → 784 features)\n",
    "x_train = x_train.reshape(-1, 784)\n",
    "x_test = x_test.reshape(-1, 784)\n",
    "\n",
    "# Define anomaly: Digit 9 = anomaly, 0–8 = normal\n",
    "normal_idx = y_train != 9\n",
    "x_train_normal = x_train[normal_idx]\n",
    "\n",
    "x_test_normal = x_test[y_test != 9]\n",
    "x_test_anomaly = x_test[y_test == 9]\n",
    "\n",
    "# Combine test data\n",
    "x_test_combined = np.concatenate([x_test_normal, x_test_anomaly])\n",
    "y_test_combined = np.concatenate([\n",
    "    np.zeros(len(x_test_normal)),   # Normal = 0\n",
    "    np.ones(len(x_test_anomaly))    # Anomaly = 1\n",
    "])\n",
    "\n",
    "print(\"Training Normal Shape:\", x_train_normal.shape)\n",
    "print(\"Test Combined Shape:\", x_test_combined.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bf68f738-b931-42bf-b17d-3ff08b8e3478",
   "metadata": {},
   "outputs": [],
   "source": [
    "latent_dim = 16\n",
    "\n",
    "# Encoder\n",
    "inputs = keras.Input(shape=(784,))\n",
    "x = layers.Dense(256, activation=\"relu\")(inputs)\n",
    "x = layers.Dense(128, activation=\"relu\")(x)\n",
    "\n",
    "z_mean = layers.Dense(latent_dim, name=\"z_mean\")(x)\n",
    "z_log_var = layers.Dense(latent_dim, name=\"z_log_var\")(x)\n",
    "\n",
    "# Reparameterization Trick (SAFE VERSION)\n",
    "def sampling(args):\n",
    "    z_mean, z_log_var = args\n",
    "    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))\n",
    "    return z_mean + tf.exp(0.5 * z_log_var) * epsilon\n",
    "\n",
    "z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])\n",
    "\n",
    "# Decoder\n",
    "decoder_input = keras.Input(shape=(latent_dim,))\n",
    "d = layers.Dense(128, activation=\"relu\")(decoder_input)\n",
    "d = layers.Dense(256, activation=\"relu\")(d)\n",
    "decoder_output = layers.Dense(784, activation=\"sigmoid\")(d)\n",
    "\n",
    "decoder = keras.Model(decoder_input, decoder_output, name=\"decoder\")\n",
    "\n",
    "# VAE Output\n",
    "outputs = decoder(z)\n",
    "\n",
    "# Define VAE model (Functional API)\n",
    "vae = keras.Model(inputs, outputs, name=\"vae\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9a029dc9-27f5-485c-82d3-e6ccf88dfd93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\JEEVAKARUNYA T\\.anaconda\\Anaconda\\envs\\vae_env\\lib\\site-packages\\keras\\src\\engine\\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n",
      "Model: \"vae\"\n",
      "__________________________________________________________________________________________________\n",
      " Layer (type)                Output Shape                 Param #   Connected to                  \n",
      "==================================================================================================\n",
      " input_3 (InputLayer)        [(None, 784)]                0         []                            \n",
      "                                                                                                  \n",
      " dense_5 (Dense)             (None, 256)                  200960    ['input_3[0][0]']             \n",
      "                                                                                                  \n",
      " dense_6 (Dense)             (None, 128)                  32896     ['dense_5[0][0]']             \n",
      "                                                                                                  \n",
      " z_mean (Dense)              (None, 16)                   2064      ['dense_6[0][0]']             \n",
      "                                                                                                  \n",
      " z_log_var (Dense)           (None, 16)                   2064      ['dense_6[0][0]']             \n",
      "                                                                                                  \n",
      " lambda_1 (Lambda)           (None, 16)                   0         ['z_mean[0][0]',              \n",
      "                                                                     'z_log_var[0][0]']           \n",
      "                                                                                                  \n",
      " decoder (Functional)        (None, 784)                  236688    ['lambda_1[0][0]']            \n",
      "                                                                                                  \n",
      " tf.__operators__.add (TFOp  (None, 16)                   0         ['z_log_var[0][0]']           \n",
      " Lambda)                                                                                          \n",
      "                                                                                                  \n",
      " tf.math.square_1 (TFOpLamb  (None, 16)                   0         ['z_mean[0][0]']              \n",
      " da)                                                                                              \n",
      "                                                                                                  \n",
      " tf.math.subtract_1 (TFOpLa  (None, 16)                   0         ['tf.__operators__.add[0][0]',\n",
      " mbda)                                                               'tf.math.square_1[0][0]']    \n",
      "                                                                                                  \n",
      " tf.math.exp (TFOpLambda)    (None, 16)                   0         ['z_log_var[0][0]']           \n",
      "                                                                                                  \n",
      " tf.math.subtract (TFOpLamb  (None, 784)                  0         ['input_3[0][0]',             \n",
      " da)                                                                 'decoder[0][0]']             \n",
      "                                                                                                  \n",
      " tf.math.subtract_2 (TFOpLa  (None, 16)                   0         ['tf.math.subtract_1[0][0]',  \n",
      " mbda)                                                               'tf.math.exp[0][0]']         \n",
      "                                                                                                  \n",
      " tf.math.square (TFOpLambda  (None, 784)                  0         ['tf.math.subtract[0][0]']    \n",
      " )                                                                                                \n",
      "                                                                                                  \n",
      " tf.math.reduce_sum_1 (TFOp  (None,)                      0         ['tf.math.subtract_2[0][0]']  \n",
      " Lambda)                                                                                          \n",
      "                                                                                                  \n",
      " tf.math.reduce_sum (TFOpLa  (None,)                      0         ['tf.math.square[0][0]']      \n",
      " mbda)                                                                                            \n",
      "                                                                                                  \n",
      " tf.math.reduce_mean_1 (TFO  ()                           0         ['tf.math.reduce_sum_1[0][0]']\n",
      " pLambda)                                                                                         \n",
      "                                                                                                  \n",
      " tf.math.reduce_mean (TFOpL  ()                           0         ['tf.math.reduce_sum[0][0]']  \n",
      " ambda)                                                                                           \n",
      "                                                                                                  \n",
      " tf.math.multiply (TFOpLamb  ()                           0         ['tf.math.reduce_mean_1[0][0]'\n",
      " da)                                                                ]                             \n",
      "                                                                                                  \n",
      " tf.__operators__.add_1 (TF  ()                           0         ['tf.math.reduce_mean[0][0]', \n",
      " OpLambda)                                                           'tf.math.multiply[0][0]']    \n",
      "                                                                                                  \n",
      " add_loss (AddLoss)          ()                           0         ['tf.__operators__.add_1[0][0]\n",
      "                                                                    ']                            \n",
      "                                                                                                  \n",
      "==================================================================================================\n",
      "Total params: 474672 (1.81 MB)\n",
      "Trainable params: 474672 (1.81 MB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "__________________________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Reconstruction Loss (using MSE - more stable than BCE for MNIST)\n",
    "reconstruction_loss = tf.reduce_mean(\n",
    "    tf.reduce_sum(tf.square(inputs - outputs), axis=1)\n",
    ")\n",
    "\n",
    "# KL Divergence Loss\n",
    "kl_loss = -0.5 * tf.reduce_mean(\n",
    "    tf.reduce_sum(\n",
    "        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),\n",
    "        axis=1\n",
    "    )\n",
    ")\n",
    "\n",
    "# Total VAE Loss\n",
    "vae_loss = reconstruction_loss + kl_loss\n",
    "\n",
    "# Add loss properly\n",
    "vae.add_loss(vae_loss)\n",
    "\n",
    "# Compile\n",
    "vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))\n",
    "\n",
    "vae.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c24d3cec-07ec-4629-952b-8b5e308c536f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "381/381 [==============================] - 10s 18ms/step - loss: 51.3370 - val_loss: 41.7257\n",
      "Epoch 2/15\n",
      "381/381 [==============================] - 7s 19ms/step - loss: 38.6135 - val_loss: 36.3601\n",
      "Epoch 3/15\n",
      "381/381 [==============================] - 6s 16ms/step - loss: 35.1968 - val_loss: 34.1395\n",
      "Epoch 4/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 33.6217 - val_loss: 33.0106\n",
      "Epoch 5/15\n",
      "381/381 [==============================] - 7s 17ms/step - loss: 32.6024 - val_loss: 32.1259\n",
      "Epoch 6/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 31.9430 - val_loss: 31.3804\n",
      "Epoch 7/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 31.3819 - val_loss: 31.1114\n",
      "Epoch 8/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 30.9790 - val_loss: 30.6817\n",
      "Epoch 9/15\n",
      "381/381 [==============================] - 6s 16ms/step - loss: 30.6213 - val_loss: 30.1981\n",
      "Epoch 10/15\n",
      "381/381 [==============================] - 7s 17ms/step - loss: 30.3034 - val_loss: 30.2440\n",
      "Epoch 11/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 30.0957 - val_loss: 29.8694\n",
      "Epoch 12/15\n",
      "381/381 [==============================] - 6s 17ms/step - loss: 29.9201 - val_loss: 30.1140\n",
      "Epoch 13/15\n",
      "381/381 [==============================] - 7s 17ms/step - loss: 29.7353 - val_loss: 29.4404\n",
      "Epoch 14/15\n",
      "381/381 [==============================] - 7s 17ms/step - loss: 29.5438 - val_loss: 29.6511\n",
      "Epoch 15/15\n",
      "381/381 [==============================] - 7s 17ms/step - loss: 29.4410 - val_loss: 29.5811\n"
     ]
    }
   ],
   "source": [
    "history = vae.fit(\n",
    "    x_train_normal,\n",
    "    x_train_normal,\n",
    "    epochs=15,\n",
    "    batch_size=128,\n",
    "    validation_split=0.1,\n",
    "    verbose=1\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5c09f055-63ff-4588-b0ed-888166a28151",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'keras.src.engine.functional.Functional'>\n"
     ]
    }
   ],
   "source": [
    "print(type(vae))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6b380b9a-543c-4200-982d-e39de79dc80a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reconstruction error calculated!\n",
      "Sample errors: [0.01480477 0.0275795  0.00609045 0.01540454 0.01914495 0.00431256\n",
      " 0.04025499 0.05768408 0.01434368 0.03788473]\n"
     ]
    }
   ],
   "source": [
    "# Get VAE reconstructions\n",
    "#vae_reconstructions = vae.predict(x_test_combined)\n",
    "\n",
    "# Direct forward pass (bypasses broken predict wrapper)\n",
    "vae_reconstructions = vae(x_test_combined, training=False).numpy()\n",
    "\n",
    "# Calculate reconstruction error (MSE per sample)\n",
    "vae_errors = np.mean(np.square(x_test_combined - vae_reconstructions), axis=1)\n",
    "\n",
    "print(\"Reconstruction error calculated!\")\n",
    "print(\"Sample errors:\", vae_errors[:10])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "9c72b0dd-5eb8-4b11-be4d-4e09debd65b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "381/381 [==============================] - 7s 14ms/step - loss: 0.0399 - val_loss: 0.0180\n",
      "Epoch 2/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0141 - val_loss: 0.0124\n",
      "Epoch 3/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0102 - val_loss: 0.0095\n",
      "Epoch 4/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0084 - val_loss: 0.0087\n",
      "Epoch 5/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0075 - val_loss: 0.0079\n",
      "Epoch 6/15\n",
      "381/381 [==============================] - 5s 13ms/step - loss: 0.0068 - val_loss: 0.0069\n",
      "Epoch 7/15\n",
      "381/381 [==============================] - 5s 14ms/step - loss: 0.0063 - val_loss: 0.0068\n",
      "Epoch 8/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0060 - val_loss: 0.0064\n",
      "Epoch 9/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0057 - val_loss: 0.0062\n",
      "Epoch 10/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0055 - val_loss: 0.0059\n",
      "Epoch 11/15\n",
      "381/381 [==============================] - 6s 15ms/step - loss: 0.0053 - val_loss: 0.0057\n",
      "Epoch 12/15\n",
      "381/381 [==============================] - 5s 13ms/step - loss: 0.0052 - val_loss: 0.0060\n",
      "Epoch 13/15\n",
      "381/381 [==============================] - 5s 14ms/step - loss: 0.0051 - val_loss: 0.0058\n",
      "Epoch 14/15\n",
      "381/381 [==============================] - 5s 14ms/step - loss: 0.0050 - val_loss: 0.0061\n",
      "Epoch 15/15\n",
      "381/381 [==============================] - 5s 14ms/step - loss: 0.0049 - val_loss: 0.0054\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x1f8ef863cd0>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Build Baseline Autoencoder\n",
    "input_ae = keras.Input(shape=(784,))\n",
    "x = layers.Dense(256, activation=\"relu\")(input_ae)\n",
    "x = layers.Dense(64, activation=\"relu\")(x)\n",
    "x = layers.Dense(256, activation=\"relu\")(x)\n",
    "output_ae = layers.Dense(784, activation=\"sigmoid\")(x)\n",
    "\n",
    "autoencoder = keras.Model(input_ae, output_ae)\n",
    "autoencoder.compile(optimizer=\"adam\", loss=\"mse\")\n",
    "\n",
    "# Train Autoencoder\n",
    "autoencoder.fit(\n",
    "    x_train_normal,\n",
    "    x_train_normal,\n",
    "    epochs=15,\n",
    "    batch_size=128,\n",
    "    validation_split=0.1,\n",
    "    verbose=1\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8dc12365-34cb-4a5b-9a98-c21d8a5df150",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 2s 4ms/step\n",
      "Autoencoder reconstruction error calculated!\n"
     ]
    }
   ],
   "source": [
    "# Autoencoder reconstructions\n",
    "ae_reconstructions = autoencoder.predict(x_test_combined)\n",
    "\n",
    "# Reconstruction error\n",
    "ae_errors = np.mean(np.square(x_test_combined - ae_reconstructions), axis=1)\n",
    "\n",
    "print(\"Autoencoder reconstruction error calculated!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "20939102-6e27-49f4-b2b2-0108f51fcfe7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "VAE AUC Score: 0.675559107174568\n",
      "Autoencoder AUC Score: 0.48828478296598543\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import roc_auc_score, roc_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# AUC Scores\n",
    "vae_auc = roc_auc_score(y_test_combined, vae_errors)\n",
    "ae_auc = roc_auc_score(y_test_combined, ae_errors)\n",
    "\n",
    "print(\"VAE AUC Score:\", vae_auc)\n",
    "print(\"Autoencoder AUC Score:\", ae_auc)\n",
    "\n",
    "# ROC Curve for VAE\n",
    "fpr, tpr, _ = roc_curve(y_test_combined, vae_errors)\n",
    "\n",
    "plt.figure(figsize=(6,5))\n",
    "plt.plot(fpr, tpr, label=f\"VAE ROC (AUC = {vae_auc:.3f})\")\n",
    "plt.xlabel(\"False Positive Rate\")\n",
    "plt.ylabel(\"True Positive Rate\")\n",
    "plt.title(\"ROC Curve - VAE Anomaly Detection\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ba7393ec-8ccb-4442-a3f0-8390fd4d7686",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 3ms/step\n",
      "Latent Space Statistics:\n",
      "Mean (μ) - Normal Data: -0.008011595\n",
      "Variance (σ²) - Normal Data: 0.5047413\n",
      "Mean (μ) - Anomaly Data: 0.043612443\n",
      "Variance (σ²) - Anomaly Data: 0.38737434\n"
     ]
    }
   ],
   "source": [
    "# Create encoder model to extract latent mean\n",
    "encoder = keras.Model(inputs, z_mean)\n",
    "\n",
    "# Get latent vectors\n",
    "z_test = encoder.predict(x_test_combined)\n",
    "\n",
    "# Separate normal and anomaly latent representations\n",
    "normal_latent = z_test[y_test_combined == 0]\n",
    "anomaly_latent = z_test[y_test_combined == 1]\n",
    "\n",
    "# Compute statistics\n",
    "print(\"Latent Space Statistics:\")\n",
    "print(\"Mean (μ) - Normal Data:\", np.mean(normal_latent))\n",
    "print(\"Variance (σ²) - Normal Data:\", np.var(normal_latent))\n",
    "print(\"Mean (μ) - Anomaly Data:\", np.mean(anomaly_latent))\n",
    "print(\"Variance (σ²) - Anomaly Data:\", np.var(anomaly_latent))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9c5955f4-f16e-4375-abf4-b279c5aa0878",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'keras.src.engine.functional.Functional'>\n"
     ]
    }
   ],
   "source": [
    "print(type(vae))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cf5b77fb-f7ef-4437-8e50-ba9169c94fd2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Reconstruction Error Distribution\n",
    "plt.figure(figsize=(8,5))\n",
    "\n",
    "plt.hist(vae_errors[y_test_combined == 0], bins=50, alpha=0.6, label=\"Normal Data\")\n",
    "plt.hist(vae_errors[y_test_combined == 1], bins=50, alpha=0.6, label=\"Anomaly (Digit 9)\")\n",
    "\n",
    "plt.xlabel(\"Reconstruction Error\")\n",
    "plt.ylabel(\"Frequency\")\n",
    "plt.title(\"Reconstruction Error Distribution - VAE\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b9e455c-d5c7-4036-92bf-6770d40e34e6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (VAE Env)",
   "language": "python",
   "name": "vae_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
