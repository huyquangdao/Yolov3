import time

class Timer:

    def __init__(self):
        pass

    def __call__(self, function):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (function.__name__,  end - start))
            return result

        return wrapper
