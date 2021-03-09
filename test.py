def gen():
    for i in range(0, 10):
        yield True, i


if __name__ == '__main__':
    for i in gen():
        print(i)