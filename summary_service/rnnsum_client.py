import argparse
import subprocess


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--qExpansion", type=str, required=True)
    parser.add_argument("--qResults", type=str, required=False)
    parser.add_argument("--experiment", type=str, required=False)
    parser.add_argument("--dataStructure", type=str, required=False)
    args = parser.parse_args()
    
    subprocess.run(["python", "summary_client.py", "--port", str(args.port),
                    "--action", "query", "--id", args.qExpansion])

if __name__ == "__main__":
    main() 
