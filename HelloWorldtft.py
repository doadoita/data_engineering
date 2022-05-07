# se hacen los import de lo que se necesitará
import os
import tempfile
import pprint

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tfx_bsl.public import tfxio

# Se recolecta la muestra de data
raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

# Se deine la metadata y el schema

raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'y': tf.io.FixedLenFeature([], tf.float32),
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
    }))

# se define la función de preprocesamiento
def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    
    # se extraen las columnas y se les asignan variables locales 
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']
    # transformaciones de la data usando funciones de tft
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    s_integerized = tft.compute_and_apply_vocabulary(s)
    x_centered_times_y_normalized = x_centered * y_normalized
    
  # El resultado es la data transformada
    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        'x_centered_times_y_normalized': x_centered_times_y_normalized,
        's_integerized': s_integerized
    }

# Es necesario un directorio temporal para analizar la data
def main():
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):

    #se define el pipepline usando Apache Beam syntax
        transformed_dataset, transform_fn = (
        
        # se analiza y transforma el dataset usando la función de preprocesamiento
        (raw_data, raw_data_metadata) |
        tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
    # el dataset transformado tiene data y metadata, por lo que se separan
    transformed_data, transformed_metadata = transformed_dataset

    # se imprimen los resultados
    print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
    print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))


if __name__ == '__main__':
    main()