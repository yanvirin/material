docker run -p $4:$4 -itd  -v $1:/app/inputs -v $2:/app/outputs -v $3:/app/query  -e "PORT=$4" --name rnnsum-v0.1-instance rnnsum:v0.1 /bin/bash
docker start rnnsum-v0.1-instance
docker attach rnnsum-v0.1-instance &
