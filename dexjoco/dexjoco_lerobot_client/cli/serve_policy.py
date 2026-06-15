import os


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

    from dexjoco_lerobot_client.policy_server import serve

    serve()


if __name__ == "__main__":
    main()
