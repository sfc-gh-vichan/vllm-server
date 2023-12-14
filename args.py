from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser(description="vLLM server")
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        help="Port to run the service on",
        default="1234",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
