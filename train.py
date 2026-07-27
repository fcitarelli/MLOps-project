from src.dataset import prepare_dataset
from src.training import train


def main():

    dataset = prepare_dataset()

    train(dataset)


if __name__ == "__main__":
    main()