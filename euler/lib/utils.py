def error(test: bool, msg: str):
    if not test:
        raise ValueError(msg)


def warning(test: bool, msg: str):
    if not test:
        print('warning: {}'.format(msg))