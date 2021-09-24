if [ $# != 1 ]; then
    echo "usage: "$0" use_rpc"
    echo "       use_rpc: {0, 1}"
    exit
fi

if [ $1 -eq 0 ]; then
    echo "not use rpc"
    nohup python poem_generation.py &
else
    echo "use rpc"
    nohup python poem_generation_rpc_server.py &
    nohup python poem_generation_rpc_client.py &
fi
