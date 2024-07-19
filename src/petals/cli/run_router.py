import argparse
from petals.router.router import Http_server

def main():
    parser = argparse.ArgumentParser(description="Run HTTP server for collecting prompts.")
    parser.add_argument('--initial_peers', type=str, nargs='+', required=True,
                        help='List of multiaddrs of initial peers')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or path for the distributed model')

    args = parser.parse_args()

    initial_peers = args.initial_peers
    model_name = args.model_name

    server = Http_server(initial_peers, model_name)
    print(server.initial_peers)
    print(server.model_name)
    # server.server_start()


if __name__ == "__main__":
    main()