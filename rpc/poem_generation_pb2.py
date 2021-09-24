# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rpc/poem_generation.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='rpc/poem_generation.proto',
  package='language',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x19rpc/poem_generation.proto\x12\x08language\"+\n\nGenRequest\x12\r\n\x05genre\x18\x01 \x01(\r\x12\x0e\n\x06prefix\x18\x02 \x01(\t\"\x18\n\x08GenReply\x12\x0c\n\x04poem\x18\x01 \x01(\t2H\n\x0ePoemGeneration\x12\x36\n\x08Generate\x12\x14.language.GenRequest\x1a\x12.language.GenReply\"\x00\x62\x06proto3'
)




_GENREQUEST = _descriptor.Descriptor(
  name='GenRequest',
  full_name='language.GenRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='genre', full_name='language.GenRequest.genre', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prefix', full_name='language.GenRequest.prefix', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=82,
)


_GENREPLY = _descriptor.Descriptor(
  name='GenReply',
  full_name='language.GenReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='poem', full_name='language.GenReply.poem', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=108,
)

DESCRIPTOR.message_types_by_name['GenRequest'] = _GENREQUEST
DESCRIPTOR.message_types_by_name['GenReply'] = _GENREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GenRequest = _reflection.GeneratedProtocolMessageType('GenRequest', (_message.Message,), {
  'DESCRIPTOR' : _GENREQUEST,
  '__module__' : 'rpc.poem_generation_pb2'
  # @@protoc_insertion_point(class_scope:language.GenRequest)
  })
_sym_db.RegisterMessage(GenRequest)

GenReply = _reflection.GeneratedProtocolMessageType('GenReply', (_message.Message,), {
  'DESCRIPTOR' : _GENREPLY,
  '__module__' : 'rpc.poem_generation_pb2'
  # @@protoc_insertion_point(class_scope:language.GenReply)
  })
_sym_db.RegisterMessage(GenReply)



_POEMGENERATION = _descriptor.ServiceDescriptor(
  name='PoemGeneration',
  full_name='language.PoemGeneration',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=110,
  serialized_end=182,
  methods=[
  _descriptor.MethodDescriptor(
    name='Generate',
    full_name='language.PoemGeneration.Generate',
    index=0,
    containing_service=None,
    input_type=_GENREQUEST,
    output_type=_GENREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_POEMGENERATION)

DESCRIPTOR.services_by_name['PoemGeneration'] = _POEMGENERATION

# @@protoc_insertion_point(module_scope)
