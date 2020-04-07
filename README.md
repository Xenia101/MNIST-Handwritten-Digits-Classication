# A Simple MNIST Handwritten Digits Classication

Handwriting Identification Using **MNIST Dataset**

```python
train_pixels,train_list_values = mnist.train.next_batch(n_train_data) 
test_pixels,test_list_of_values = mnist.test.next_batch(n_test_data) 

train_pixel_tensor = tf.placeholder("float",[None,784])
test_pixel_tensor = tf.placeholder("float",[784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.negative(test_pixel_tensor))),reduction_indices=1)
```

## Example

```python
predicted number :  4 => answer :  4
predicted number :  3 => answer :  8
predicted number :  6 => answer :  6
predicted number :  6 => answer :  3
predicted number :  1 => answer :  1
predicted number :  3 => answer :  5
predicted number :  5 => answer :  5
predicted number :  2 => answer :  2
predicted number :  0 => answer :  0
predicted number :  1 => answer :  4
Accuracy =  0.6
```

## Execution / Test Environment
  - Windows 10 or Ubuntu Linux
  - Python **3.6**

## Usage
- Input : ```./mnist_data```

  `python3 Digits Classication.py`

- Output : Accuracy
