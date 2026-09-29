import logging

from cdl.main import training_pipeline


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    training_pipeline()


if __name__ == "__main__":
    main()
