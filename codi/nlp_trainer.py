"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from pickle import dump, load
import tensorflow as tf
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from codi.mlp_codi import MLPCoDi
from codi.codi_utils import generate_plots, \
                            plot_precision_threshold_tradeoff, \
                            filter_nlp_datapoints

from codi.codi_trainer import CoDiTrainer


class NLPTrainer(MLPCoDi, CoDiTrainer):
    """
    Creates a CoDiTrainer instance.

    Parameters:
        name: string
        yaml_model_path: path of yaml containing model params.
        yaml_train_path: path to yaml containing training params.

    Attributes:
        datasets: train_dataset, test_dataset, validation_dataset, unlabelled_dataset
        scaler: the scaler instance ued to scale train, val and test sets.
        train_params: parameters extracted from yaml
        class_weights: class weights computed for imbalanced classes.
    """
    def __init__(self, name='name', yaml_model_path='mlp_codi.yaml',
                 yaml_train_path='train_codi.yaml'):
        super(NLPTrainer, self).__init__(name=name, yaml_path=yaml_model_path)
        self.load_model()
        self.train_params = self.load_yaml(yaml_train_path)
        self.scaler = None
        self.class_weights = None
        # all the following functions are updated in create_labelled_Dataset.
        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.unlabelled_dataset = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def train(self):
        """
        Trains the model.
        """
        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=self.train_params['patience'],
                                           mode='auto',
                                           baseline=None, restore_best_weights=True)
        self.compile(loss=self.train_params['loss'], optimizer=self.train_params['optimizer'],
                     metrics=['accuracy'])

        history = self.fit(self.train_dataset, validation_data=self.validation_dataset,
                           epochs=self.train_params['epochs'],
                           class_weight=self.class_weights, callbacks=[es])

        test_score = self.evaluate(self.test_dataset)

        self.save_weights(self.train_params['path_to_ckpt'])

        test_pred = self.predict(self.test_dataset)

        #generate_plots([history.history['accuracy'], history.history['loss'],
        #               history.history['val_loss'], history.history['val_accuracy']])
        #plot_precision_threshold_tradeoff(test_pred, self.y_test)

        return history.history['accuracy'][-1], test_score[-1]

    def create_labelled_dataset(self, iterator):
        """
        Creates labelled dataset as tf.data.dataset object

        Args:
            iterator: A customed torch iterator.
        """
        features = self.train_params['features']

        for idx, example_batch in enumerate(iterator):
            y_batch = [example.is_correct for example in example_batch[0]]
            X_batch = np.empty((len(example_batch[0]), 0))
            for feature in features:
                feat_arr = [getattr(example, feature) for example in example_batch[0]]
                if len(np.shape(feat_arr)) == 3:
                    feat_arr = np.squeeze(feat_arr, axis=2)
                if len(np.shape(feat_arr)) == 1:
                    feat_arr = np.expand_dims(feat_arr, axis=1)
                X_batch = np.concatenate([X_batch, feat_arr], axis=1)
            if idx == 0:
                X = X_batch
                y = y_batch
            else:
                X = np.append(X, X_batch, axis=0)
                y = np.append(y, y_batch, axis=0)

        y = y.astype(np.float32)

        self.scaler = StandardScaler()

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.1,
                                                                              random_state=42)
        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                              np.unique(self.y_train),
                                                                              self.y_train)))
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=512).batch(self.train_params['batch_size'])

        self.validation_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.validation_dataset = self.validation_dataset.batch(self.train_params['test_batch_size'])

        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.test_dataset = self.test_dataset.batch(self.train_params['test_batch_size'])

        dump(self.scaler, open(self.train_params['path_to_scaler'], 'wb'))

    def create_unlabelled_dataset(self, iterator):
        """
        Creates unlabelled dataset as tf.data.dataset object.

        Args:
            iterator: A customed torch iterator.
        """
        features = self.train_params['features']

        for example_batch, _ in iterator:
            X_batch = np.empty((len(example_batch), 0))
            for feature in features:
                feat_arr = [getattr(example, feature) for example in example_batch]
                if len(np.shape(feat_arr)) == 3:
                    feat_arr = np.squeeze(feat_arr, axis=2)
                if len(np.shape(feat_arr)) == 1:
                    feat_arr = np.expand_dims(feat_arr, axis=1)
                X_batch = np.concatenate([X_batch, feat_arr], axis=1)

            self.scaler = load(open(self.train_params['path_to_scaler'], 'rb'))

            X_batch = self.scaler.transform(X_batch)
            self.unlabelled_dataset = tf.data.Dataset.from_tensor_slices(X_batch)

            self.unlabelled_dataset = self.unlabelled_dataset.batch(len(X_batch))

            scores = self.test()
            for index, example in enumerate(example_batch):
                example.score = scores[index]

        return filter_nlp_datapoints(iterator, self.train_params['filtering'])

    def test(self):
        """
        Classifies (evaluates) new unlabelled data using the previously trained model.
        """
        self.load_weights(self.train_params['path_to_ckpt']).expect_partial()
        return self.predict(self.unlabelled_dataset)
