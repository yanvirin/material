import argparse
import subprocess


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--qExpansion", type=str, required=True)
    parser.add_argument("--qResults", type=str, required=False)
    parser.add_argument("--experiment", type=str, required=False)
    parser.add_argument("--dataStructure", type=str, required=False)

    parser.add_argument("--waitTime", required=False, type=int, default=30)
    parser.add_argument("--maxWaitAttempts", required=False, type=int, 
                        default=60)
    args = parser.parse_args()
    subprocess.run([
        "python", "/libs/material/summary_service/summary_client.py", 
        "--port", str(args.port), "--action", "query", "--id", args.qExpansion,
        "--waitTime", str(args.waitTime), "--maxWaitAttempts",
        str(args.maxWaitAttempts)])

if __name__ == "__main__":
    main() 
