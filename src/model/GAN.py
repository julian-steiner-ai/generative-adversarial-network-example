"""
Implementation of a Generative Adversarial Network.
"""

from os.path import join

from loguru import logger

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.initializers import RandomNormal

import numpy as np
import pickle as pkl

class GAN:
    @staticmethod
    def load(directory):
        with open(join(directory, 'params.pkl'), 'rb') as f:
            params = pkl.load(f)
        gan = GAN(*params)
        gan.load_weights(join(directory, 'weights/weights.h5'))
        return gan;

    def __init__(self,
                 input_dim,
                 discriminator_conv_filters,
                 discriminator_conv_kernel_size,
                 discriminator_conv_strides,
                 discriminator_batch_norm_momentum,
                 discriminator_activation,
                 discriminator_dropout_rate,
                 discriminator_learning_rate,
                 generator_initial_dense_layer_size,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernel_size,
                 generator_conv_strides,
                 generator_batch_norm_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 optimizer,
                 z_dim):
        
        self.name = 'GAN'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        
        self.optimizer_str = optimizer
        self.z_dim = z_dim

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.discriminator_losses = []
        self.generator_losses = []

        self.epoch = 0

        self._init_discriminator()
        self._init_generator()
        self._init_adversarial()

    def _get_activiation_layer(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer

    def _init_discriminator(self):
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input

        for i in range(self.n_layers_discriminator):
            x = Conv2D(
                filters=self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernel_size[i],
                strides=self.discriminator_conv_strides[i],
                padding='same',
                name='discriminator_conv' + str(i),
                kernel_initializer=self.weight_init
            )(x)

            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)

            x = self._get_activiation_layer(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(rate = self.discriminator_dropout_rate)(x)
        
        x = Flatten()(x)

        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)

        self.discriminator = Model(discriminator_input, discriminator_output)
    
    def _init_generator(self):
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input

        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

        x = self._get_activiation_layer(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):

            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters = self.generator_conv_filters[i],
                    kernel_size = self.generator_conv_kernel_size[i],
                    padding = 'same',
                    name = 'generator_conv' + str(i),
                    kernel_initializer = self.weight_init
                )(x)
            else:
                x = Conv2DTranspose(
                    filters = self.generator_conv_filters[i],
                    kernel_size = self.generator_conv_kernel_size[i],
                    padding = 'same',
                    strides = self.generator_conv_strides[i],
                    name = 'generator_conv' + str(i),
                    kernel_initializer = self.weight_init
                )(x)

            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

                x = self._get_activiation_layer(self.generator_activation)(x)
            else:

                x = Activation('tanh')(x)

        generator_output = x

        self.generator = Model(generator_input, generator_output)

    def _get_optimizer_layer(self, lr):
        if self.optimizer_str == 'adam':
            return Adam(learning_rate=lr, beta_1=0.5)
        elif self.optimizer_str == 'rmsprop':
            return RMSprop(learning_rate=lr)
        else:
            return Adam(learning_rate=lr)

    def _set_trainable(self, model, value):
        model.trainable = value
        for layer in model.layers:
            layer.trainable = value

    def _init_adversarial(self):
        self.discriminator.compile(
            optimizer=self._get_optimizer_layer(self.discriminator_learning_rate),
            loss='binary_crossentropy',
            metrics = ['accuracy']
        )
        
        self._set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))

        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self._get_optimizer_layer(self.generator_learning_rate) , loss='binary_crossentropy', metrics=['accuracy'])

        self._set_trainable(self.discriminator, True)

    def _train_discriminator(self,
                             x_train,
                             batch_size,
                             using_generator):
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real =   self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return {
            'd_loss': d_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake,
            'd_acc': d_acc,
            'd_acc_real': d_acc_real,
            'd_acc_fake': d_acc_fake
        }

    def _train_generator(self,
                         batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self,
              x_train,
              batch_size,
              epochs,
              print_every_n_batches=50,
              using_generator=False,
              log=False):
        for epoch in range(self.epoch, self.epoch + epochs):
            discriminator = self._train_discriminator(x_train, batch_size, using_generator)
            generator = self._train_generator(batch_size)

            if log:
                logger.info(f"{epoch} [D loss: ({discriminator['d_loss']:.3f})(R {discriminator['d_loss_real']:.3f}, F {discriminator['d_loss_fake']:.3f})] [D acc: ({discriminator['d_acc']:.3f})({discriminator['d_acc_real']:.3f}, {discriminator['d_acc_fake']:.3f})] [G loss: {generator[0]:.3f}] [G acc: {generator[1]:.3f}]")

            self.discriminator_losses.append(discriminator)
            self.generator_losses.append(generator)

            if epoch % print_every_n_batches == 0:
                logger.info(f"{epoch} [D loss: ({discriminator['d_loss']:.3f})(R {discriminator['d_loss_real']:.3f}, F {discriminator['d_loss_fake']:.3f})] [D acc: ({discriminator['d_acc']:.3f})({discriminator['d_acc_real']:.3f}, {discriminator['d_acc_fake']:.3f})] [G loss: {generator[0]:.3f}] [G acc: {generator[1]:.3f}]")
                self.model.save_weights(join('./model', 'weights', 'weights-%d.h5' % (epoch)))
                self.model.save_weights(join('./model', 'weights', 'weights.h5'))
                self.save()

            self.epoch += 1
        
    def generate(self, n_images):
        noise = np.random.normal(0, 1, (n_images, self.z_dim))
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * (generated_images + 1)
        generated_images = np.clip(generated_images, 0, 1)

        return generated_images
    
    def save(self):
        self.model.save(join('./model', 'model.h5'))
        self.discriminator.save(join('./model', 'discriminator.h5'))
        self.generator.save(join('./model', 'generator.h5'))
        self._save_params()
        pkl.dump(self, open(join('./model', "obj.pkl"), "wb"))

    def _save_params(self):
        with open(join('./model', 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim,
                self.discriminator_conv_filters,
                self.discriminator_conv_kernel_size,
                self.discriminator_conv_strides,
                self.discriminator_batch_norm_momentum,
                self.discriminator_activation,
                self.discriminator_dropout_rate,
                self.discriminator_learning_rate,
                self.generator_initial_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate,
                self.optimizer_str,
                self.z_dim], f)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)