from pathlib import Path
from typing import (
    List,
    Dict,
    Union,
    Tuple
)
import numpy as np
import tensorflow.keras
from keras.utils import np_utils
from model import build_model
from copy import deepcopy as dc
from datetime import datetime


def get_timestamp() -> str:
    """

    Returns:

    """
    return "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())


def character_map(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """

    Args:
        text:

    Returns:

    """
    chars = sorted(list(set(text)))
    char_to_num = {item: idx for idx, item in enumerate(chars)}
    num_to_char = {idx: item for idx, item in enumerate(chars)}

    return char_to_num, num_to_char


class PrepareModel:
    """

    """
    def __init__(self,
                 corpus_path: Path,
                 seq_length: int = 4,
                 numerical: bool = True
                 ):
        """

        Args:
            corpus_path:
            seq_length:
            numerical:
        """
        self.numerical = numerical
        self.seq_length = seq_length
        self.corpus_path = corpus_path
        self.corpus = self.get_corpus()
        self.char_to_num, self.num_to_char = character_map(self.corpus)
        self.num_characters = len(set(self.corpus))

    def get_corpus(self) -> str:
        """

        Returns:

        """
        with open(self.corpus_path, 'r') as fl:
            contents = fl.read()
        return contents.lower()

    def create_dataset(self) -> Tuple[List[List[Union[str, int]]], List[List[Union[str, int]]]]:
        """

        Returns:

        """
        dataset_inputs = []
        dataset_outputs = []

        stream = iter(self.corpus[self.seq_length:])
        current_y = next(stream)

        if self.numerical:
            current_x = [self.char_to_num[x] for x in self.corpus[:self.seq_length]]
        else:
            current_x = [x for x in self.corpus[:self.seq_length]]

        while True:
            try:
                dataset_inputs.append(dc(current_x))
                to_append_output = current_y
                if self.numerical:
                    to_append_output = self.char_to_num[current_y]
                dataset_outputs.append([to_append_output])

                current_x = dc(dataset_inputs[-1])
                current_x.pop(0)
                current_x.append(to_append_output)
                current_y = next(stream)
            except StopIteration as e:
                print("Done!")
                break
        return dataset_inputs, dataset_outputs

    def transform_into_vectors(self, inputs: List[List[int]], outputs: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            inputs:
            outputs:

        Returns:

        """
        inputs_modified = np.reshape(inputs, (len(inputs), self.seq_length, 1))
        inputs_modified = inputs_modified / float(self.num_characters)
        outputs_modified = np_utils.to_categorical(outputs)
        return inputs_modified, outputs_modified

    def generate(self,
                 model: tensorflow.keras.models.Sequential,
                 starting_prompt: str,
                 number_of_characters: int
                 ) -> str:
        """

        Args:
            model:
            starting_prompt:
            number_of_characters:

        Returns:

        """
        assert len(starting_prompt) >= self.seq_length

        encoded = [self.char_to_num[x] for x in starting_prompt[-self.seq_length:]]
        encoded_results = dc(encoded)
        for _ in range(number_of_characters):
            input_vector = np.array(encoded, dtype='float64')
            input_vector *= (1. / float(self.num_characters))
            input_vector = np.reshape(input_vector, (1, self.seq_length, 1))

            prediction_index = np.argmax(
                model.predict(input_vector, verbose=0)
            ).item()
            encoded_results.append(prediction_index)
            encoded.pop(0)
            encoded.append(prediction_index)

        decoded_results = [self.num_to_char[n] for n in encoded_results]
        return "".join(decoded_results)

    def train(self,
              mdl: tensorflow.keras.models.Sequential,
              **kwargs
              ) -> tensorflow.keras.models.Sequential:
        """

        Args:
            mdl:
            **kwargs:

        Returns:

        """
        mdl.compile(loss='categorical_crossentropy', optimizer='adam')
        given_inputs, expected_outputs = self.transform_into_vectors(*self.create_dataset())
        mdl.fit(given_inputs,
                expected_outputs,
                epochs=kwargs.get('epochs'),
                batch_size=kwargs.get('batch_size'),
                validation_split=.3
                )
        filepath = kwargs.get('save_path')
        mdl.save_weights(filepath)
        return mdl.load_weights(filepath)


def train_with_setup(**kwargs):
    pm = PrepareModel(
        corpus_path=kwargs.get('corpus_path'),
        seq_length=kwargs.get('seq_length')
    )
    mdl = build_model(input_size=pm.seq_length,
                      output_size=pm.num_characters
                      )
    train_config = {
        'batch_size': kwargs.get('batch_size'),
        'epochs': kwargs.get('epochs')
    }
    pm.train(mdl, **train_config)


def generate_with_setup(**kwargs):
    pm = PrepareModel(
        corpus_path=kwargs.get('corpus_path'),
        seq_length=kwargs.get('seq_length'),
    )
    mdl = build_model(input_size=pm.seq_length,
                      output_size=pm.num_characters
                      )
    mdl_path = kwargs.get('model_trained')
    mdl.load_weights(mdl_path)
    print(pm.generate(mdl, pm.corpus[100: 100 + pm.seq_length], kwargs.get('characters')))
    return


def main():
    config = {
        'seq_length': 10,
        'batch_size': 30,
        'epochs': 2,
        'corpus_path': Path('./data/eilish_combined.txt'),
        # 'save_path' : Path(f'./models/W-AI-lde_{get_timestamp()}.h5'),
        'characters': 1000,
        'model_trained': Path('./models/AIlish_2021_07_19_21_43_36.h5')
    }
    # train_with_setup(**config)
    generate_with_setup(**config)
    pass


if __name__ == '__main__':
    main()
