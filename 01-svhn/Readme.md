    '''
    Homework_1.md中Lp-pooling的实现方法
    在model.py中修改
    '''
    def _pool_layer(self, name, inp, ksize, stride, padding='SAME', mode='MAX', p=None):
        assert mode in ['MAX', 'AVG', 'LP'], 'the mode of pool must be MAX or AVG'
        if p is not None:
            assert isinstance(p, int), 'p must be integer!'
            assert mode == 'LP'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride],
                               padding=padding, name=name, data_format='NCHW')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride],
                               padding=padding, name=name, data_format='NCHW')
        elif mode  == 'LP':
            x = tf.pow(inp, p)
            x = tf.nn.avg_pool(x, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride],
                               padding=padding, name=name, data_format='NCHW')
            x = tf.pow(inp, 1.0 / p)
        return x
