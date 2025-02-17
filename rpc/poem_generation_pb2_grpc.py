# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from rpc import poem_generation_pb2 as rpc_dot_poem__generation__pb2


class PoemGenerationStub(object):
    """The poem generation service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Generate = channel.unary_unary(
                '/language.PoemGeneration/Generate',
                request_serializer=rpc_dot_poem__generation__pb2.GenRequest.SerializeToString,
                response_deserializer=rpc_dot_poem__generation__pb2.GenReply.FromString,
                )


class PoemGenerationServicer(object):
    """The poem generation service definition.
    """

    def Generate(self, request, context):
        """Generate a poem
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PoemGenerationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Generate': grpc.unary_unary_rpc_method_handler(
                    servicer.Generate,
                    request_deserializer=rpc_dot_poem__generation__pb2.GenRequest.FromString,
                    response_serializer=rpc_dot_poem__generation__pb2.GenReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'language.PoemGeneration', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PoemGeneration(object):
    """The poem generation service definition.
    """

    @staticmethod
    def Generate(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/language.PoemGeneration/Generate',
            rpc_dot_poem__generation__pb2.GenRequest.SerializeToString,
            rpc_dot_poem__generation__pb2.GenReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
