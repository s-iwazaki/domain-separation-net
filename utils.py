import tensorflow as tf

def get_dataset(file_path, batch_size=10000, is_shuffle=True, epochs=None)
    ds = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_size,
      num_epochs=epochs,
      shuffle=is_shuffle
      na_value="?",
      field_delim='\t',
      ignore_errors=True)
    
    return ds