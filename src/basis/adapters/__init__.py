"""Consumer adapters.

basis keeps its own vocabulary (``tenant_id``, its own tables). An adapter maps
that vocabulary onto a specific consumer's existing schema, so basis stays
portable and the consumer keeps the schema it already has.

Adapters are imported explicitly - importing this package pulls in nothing.
"""

__all__: list[str] = []
