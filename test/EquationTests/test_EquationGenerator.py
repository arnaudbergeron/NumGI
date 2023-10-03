from __future__ import annotations

import pytest

from NumGI.SolutionGenerator import SolutionGenerator


@pytest.fixture
def tree():
    tree = SolutionGenerator.EquationTree(
        SolutionGenerator.EquationTree.Node(("function", "function"), None, None, 0)
    )
    return tree


def test_insert_tree(
    tree,
):
    tree.insert(
        tree.root,
        SolutionGenerator.EquationTree.Node(("differential", "differential"), None, None, 0),
        SolutionGenerator.EquationTree.Node(("symbol", "symbol"), None, None, 1),
    )
    assert tree.root.right.level == 1
    assert tree.root.left.level == 1
    assert tree.level == 1

    old_node = tree.get_nodes_at_level(1)[0]
    tree.insert(
        old_node,
        SolutionGenerator.EquationTree.Node(("integration", "integration"), None, None, 1),
        SolutionGenerator.EquationTree.Node(("symbol", "symbol"), None, None, 2),
    )
    assert tree.level == 2
    assert tree.root.right.right.level == 2
    assert tree.root.right.left.level == 2
