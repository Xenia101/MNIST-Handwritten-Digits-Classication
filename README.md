# A Simple MNIST Handwritten Digits Classication

Handwriting Identification Using MNIST Dataset

```python
train_pixels,train_list_values = mnist.train.next_batch(n_train_data) 
test_pixels,test_list_of_values = mnist.test.next_batch(n_test_data) 

train_pixel_tensor = tf.placeholder("float",[None,784])
test_pixel_tensor = tf.placeholder("float",[784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.negative(test_pixel_tensor))),reduction_indices=1)
```

## 설치 방법
- 실행 환경 (테스트 환경)
  - Windows 10 or Ubuntu Linux
  - Python3.x
  
## 사용 방법
- Input : ```./mnist_data```

  `python3 Digits Classication.py`

- Output : Accuracy
