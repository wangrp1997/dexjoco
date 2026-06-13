import os

import tyro


def main():
    for proxy_var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(proxy_var, None)

    from dexjoco_lerobot_client.eval_dexjoco_lerobot import main as evaluate_dexjoco_lerobot

    tyro.cli(evaluate_dexjoco_lerobot)


if __name__ == "__main__":
    main()
