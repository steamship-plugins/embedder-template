from steamship.plugin.service import PluginRequest
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from src.api import EmbedderPlugin
from steamship.data.block import Block
from steamship.data.file import File

import os
from typing import List

__copyright__ = "Steamship"
__license__ = "MIT"

def _get_test_facts() -> List[Block]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'facts.txt'), 'r') as f:
        return list(map(lambda text: Block(text=text), f.read().split("\n")))

def dist(e1: List[float], e2: List[float]) -> float:
    """A very simplistic vector distance calculation: sum element-wise deviation. Lower means more similar."""
    ret = 0.0
    for (f1, f2) in zip(e1, e2):
        ret += abs(f1 - f2)
    return ret

def search(embeddings: List[List[float]], query: List[float]) -> int:
    """Returns the index of the embedding in `embeddings` that has the lowest `dist` score to `query`."""
    best_index = -1
    best_score = 1000

    for (i, e) in enumerate(embeddings):
        score = dist(e, query)
        if score < best_score:
            best_score = score
            best_index = i

    return best_index

def test_embedder():
    """Tests that our embedder properly associates certain sentences nearby known facts in embedding space."""
    embedder = EmbedderPlugin()

    facts = _get_test_facts()
    request = PluginRequest(data=BlockAndTagPluginInput(file=File(blocks=facts)))
    response = embedder.run(request)
    assert(response.error is None)
    assert(response.data is not None)

    embeddings = response.data.embeddings

    tests = [
        ("What does Ted think about eggs?", "Ted thinks eggs are good."),
        ("Can everyone eat cake?", "Armadillos are allergic to cake."),
        ("Can armadillos eat everything?", "Armadillos are allergic to cake."),
        ("Who should I give this apple to?", "Jerry likes to eat apples.")
    ]

    for test in tests:

        request = PluginRequest(data=BlockAndTagPluginInput(file=File(blocks=[Block(text=test[0])])))
        query_response = embedder.run(request)

        assert (query_response.error is None)
        assert (query_response.data is not None)
        assert (query_response.data.embeddings is not None)
        assert (len(query_response.data.embeddings) == 1)

        query = query_response.data.embeddings[0]
        idx = search(embeddings, query)
        assert(idx >= 0)
        assert(idx < len(facts))
        assert(facts[idx].text == test[1])

