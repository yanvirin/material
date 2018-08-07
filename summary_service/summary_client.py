import argparse
import json
import socket
import pathlib
import time


def read_example(path):
    with open(path, "r", encoding="utf8") as fp:
        query_line = fp.readline() 
        if not query_line.startswith("Query: "):
            raise Exception("Bad line: {}".format(query_line))
        query = query_line[7:].strip()
        relevant_line = fp.readline()
        if not relevant_line.startswith("Relevant: "):
            raise Exception("Bad line: {}".format(relevant_line))
       
        relevance = int(relevant_line[10:])
        if relevance not in [0, 1]:
            raise Exception("Bad relevance value: {}".format(relevance))

        domain_line = fp.readline()
        if not domain_line.startswith("Domain: "):
            raise Exception("Bad line: {}".format(domain_line))
        domain = domain_line[8:].strip()

        explanation_line = fp.readline()
        explanation = []
        if explanation_line.strip() != "Explanation:":
            raise Exception("Bad line: {}".format(explanation_line))
        explanation.append(fp.readline().strip())
        explanation.append(fp.readline().strip())

        summary_line = fp.readline()
        if summary_line.strip() != "Summary:":
            raise Exception("Bad line: {}".format(summary_line))
        summary = []
        for line in fp:
            summary.append(line.strip())

        return {
            "query": query,
            "summary_lines": summary}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--action", type=str, required=True, 
        choices=["query", "reload", "lda", "example", "topic-cache",
                 "logging"])
    parser.add_argument("--id", type=str, required=False, default=None)
    parser.add_argument(
        "--module-name", type=str, required=False, default=None)
    parser.add_argument("--topic-model-path", type=str, default=None)
    parser.add_argument(
        "--use-topic-model", choices=[True, False], required=False, 
        default=None, 
        type=lambda x: x == "True" if x in ["True", "False"] else None)
    parser.add_argument(
        "--max-topic-words", required=False, type=int, default=None)
    parser.add_argument("--example-input", required=False, default=None)
    parser.add_argument("--example-output", required=False, default=None)
    parser.add_argument(
        "--topic-cache-action", choices=["save"], default=None, type=str)
    parser.add_argument(
        "--logging-level", type=str, default=None, required=False,
        choices=["info", "warning", "debug"])
    parser.add_argument("--waitTime", required=False, type=int, default=30)
    parser.add_argument(
        "--maxWaitAttempts", required=False, type=int, default=60)
    args = parser.parse_args()

    if args.action == "query":  
        message = {"message_type": "query",
                   "message_data": {"query_id": args.id}}  
    elif args.action == "reload":  
        message = {"message_type": "reload_module",
                   "message_data": {"module": args.module_name}}  
    elif args.action == "lda":  
        if args.topic_model_path:
            path = str(pathlib.Path(args.topic_model_path).resolve())
        else:
            path = None
        message_data = {
            "path": path,
            "use_topic_model": args.use_topic_model,
            "max_topic_words": args.max_topic_words}
        message = {"message_type": "lda",
                   "message_data": message_data}
    elif args.action == "topic-cache":  
        message_data = {"action": args.topic_cache_action}
        message = {"message_type": "topic_cache",
                   "message_data": message_data}
    elif args.action == "example":
        example = read_example(args.example_input)
        example["output_path"] = args.example_output
        message = {"message_type": "example",
                   "message_data": example}

    elif args.action == "logging":
        if args.logging_level:
            logging_level = args.logging_level.upper() 
        else: 
            logging_level = None
        message_data = {"level": logging_level}
        message = {"message_type": "logging",
                   "message_data": message_data}

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    tries = 0
    success = False
    while (not success and tries < args.maxWaitAttempts):
        try:
            s.connect(("0.0.0.0", args.port))
            success = True
        except (RuntimeError, ConnectionRefusedError):
            tries += 1
            time.sleep(args.waitTime)
            if tries == args.maxWaitAttempts:
                sys.exit(1)
                print("Failed to connect within the specified timeframe %d wait time x %d tries" % (args.waitTime,args.maxWaitAttempts))

    message_bytes = json.dumps(message).encode("utf8")
    s.send(message_bytes)
        
    msg = s.recv(10)
    print(msg.decode("utf8"))

if __name__ == "__main__":
    main()
