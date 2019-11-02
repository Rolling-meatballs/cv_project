import time

# 带有时间的打印函数，可以替代print
def log(*args, **kwargs):
    # return
    time_format = '%Y/%m/%d %H:%M:%S'
    value = time.localtime(int(time.time()))
    formatted = time.strftime(time_format, value)
    print(formatted, *args, **kwargs)