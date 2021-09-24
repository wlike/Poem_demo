"""The Python implementation of the gRPC poem generation server."""

from concurrent import futures
import logging
import sys

import grpc

from rpc import poem_generation_pb2
from rpc import poem_generation_pb2_grpc

from jiuge_lvshi import Poem

model_path = 'model_jl/epoch=0-step=49999.ckpt'
model_config = 'os_model_ch_poem/config.json'
vocab_file = 'os_model_ch_poem/vocab.txt'

class PoemGenerationServicer(poem_generation_pb2_grpc.PoemGenerationServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
        self.poem = Poem(model_path, model_config, vocab_file)

    def Generate(self, request, context):
        if request.prefix == '':
        #if not request.has_prefix() or (request.has_prefix() and request.prefix() == ''):
            prefix = None
        else:
            prefix = request.prefix
        content = self.poem.generate(genre=request.genre, prefix=prefix)
        return poem_generation_pb2.GenReply(poem=content)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    poem_generation_pb2_grpc.add_PoemGenerationServicer_to_server(
        PoemGenerationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
