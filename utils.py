import tensorflow as tf

def get_dataset(file_path, batch_size, labels)
    ds = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_size,
      label_name=labels,
      na_value="?",
      field_delim='\t',
      ignore_errors=True)
    
    return ds