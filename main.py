from os import O_ASYNC
from Predictor.engine import Predictor, read_config


def main():
    engine = Predictor(configs=read_config(config_path="./configs.yaml"))
    engine.run()


if __name__ == "__main__":
    main();