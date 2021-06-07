import koras
from koras_examples import (
    mlp_toy_problem,
    mnist,
    mnist_plot
)


def main(module):
    module.main(koras)


if __name__ == '__main__':
    main(mnist_plot)
