"""Example Steamship Embedder Plugin.

In Steamship, **Embedders** are responsible for transforming text into vectors that contain
meaning in some abstract space.

"""

from typing import List
from steamship.app import App, Response, post, create_handler
from steamship.plugin.service import PluginResponse, PluginRequest
from steamship.base.error import SteamshipError
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.embedded_items_plugin_output import EmbeddedItemsPluginOutput
from steamship.plugin.embedder import Embedder


FEATURES = ["employee", "roses", "run", "bike", "ted", "grace", "violets", "sugar", "sweet", "cake",
            "flour", "chocolate", "vanilla", "flavors", "flavor", "can", "armadillo", "pizza",
            "ship", "run", "jerry", "ronaldo", "ted", "brenda", "susan", "sam", "jerry", "shoe",
            "cheese", "water", "drink", "glass", "egg", "sleep", "code", "jonathan", "dolphin", "apple",
            "orange", "number", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
            "16", "17", "bad", "terrible", "average", "good"]

DIMENSIONALITY = len(FEATURES)

def _embed(s: str) -> List[float]:
    s = s.lower()
    return list(map(lambda word: 1.0 if word in s else 0.0, FEATURES))

class EmbedderPlugin(Embedder, App):
    """"Example Steamship Embedder plugin."""

    def run(self, request: PluginRequest[BlockAndTagPluginInput] = None) -> PluginResponse[EmbeddedItemsPluginOutput]:
        """Every plugin implements a `run` function.

        This template plugin does an extremely simple form of embedding by returning a
        vector representing which keywords (from the FEATURES list) are contained in the provided text.

        A real-world Steamship Embedder is likely to either:
        - Indirect into an embedding API hosted elsewhere. (The plugin provides adaptation, rather than inference).
        - Provide embeddings retrieved from a large neural model
        """
        if request is None:
            return Response(error=SteamshipError(message="Missing request"))
        if request.data is None:
            return Response(error=SteamshipError(message="Missing data in request"))
        if request.data.file is None:
            return Response(error=SteamshipError(message="Missing file in request data"))
        if request.data.file.blocks is None:
            return Response(error=SteamshipError(message="Missing blocks in request data"))

        embeddings = list(map(lambda s: _embed(s.text), request.data.file.blocks))
        return PluginResponse(data=EmbeddedItemsPluginOutput(embeddings=embeddings))

    @post('/embed')
    def embed(self, **kwargs) -> Response:
        """App endpoint for our plugin.

        The `run` method above implements the Plugin interface.
        This `embed` method exposes it over an HTTP endpoint as a Steamship App.

        When developing your own plugin, you can almost always leave the below code unchanged.
        """
        request = Embedder.parse_request(request=kwargs)
        response = self.run(request)
        response_dict = Embedder.response_to_dict(response)
        return Response(json=response_dict)


handler = create_handler(EmbedderPlugin)
