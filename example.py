import koras
from koras_examples import (
    mlp_toy_problem,
    mnist,
    mnist_plot,
    mnist_early_stopping,
    sin_rnn,
    imdb_birnn
)


def main(module):
    module.main(koras)


if __name__ == '__main__':
    main(imdb_birnn)
