import pytest
from software_main import TextGraphOptimized

def test_multiple_bridge_words():
    """存在多个桥接词"""
    graph = TextGraphOptimized()
    graph.graph = {
        "a": {"b": 1, "c": 1},
        "b": {"d": 1},
        "c": {"d": 1},
    }
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "d")
    assert sorted(bridges) == ["b", "c"]

def test_single_bridge_word():
    """存在单个桥接词"""
    graph = TextGraphOptimized()
    graph.graph = {
        "a": {"b": 1},
        "b": {"c": 1},
    }
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "c")
    assert bridges == ["b"]

def test_no_bridge_words():
    """无桥接词"""
    graph = TextGraphOptimized()
    graph.graph = {
        "a": {"b": 1},
        "b": {"c": 1},
        "c": {"d": 1},
    }
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "d")
    assert bridges == []

def test_word1_not_exist():
    """word1不在图中"""
    graph = TextGraphOptimized()
    graph.graph = {"b": {"c": 1}}
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "c")
    assert bridges is None

def test_word2_not_exist():
    """word2不在图中"""
    graph = TextGraphOptimized()
    graph.graph = {"a": {"b": 1}}
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "c")
    assert bridges is None

def test_case_insensitive():
    """输入大小写混合"""
    graph = TextGraphOptimized()
    graph.graph = {"apple": {"banana": 1}, "banana": {"cherry": 1}}
    graph._build_full_index()
    bridges = graph.get_bridge_words("Apple", "Cherry")
    assert bridges == ["banana"]

def test_direct_edge_no_bridge():
    """直接相连但无桥接词"""
    graph = TextGraphOptimized()
    graph.graph = {"a": {"d": 1}, "d": {}}
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "d")
    assert bridges == []

def test_both_words_not_exist():
    """word1和word2均不在图中"""
    graph = TextGraphOptimized()
    graph.graph = {"x": {"y": 1}}
    graph._build_full_index()
    bridges = graph.get_bridge_words("a", "b")
    assert bridges is None

def test_empty_loop():
    """测试空循环（没有后继节点）"""
    graph = TextGraphOptimized()
    graph.graph = {
        "apple": {"bridge1": 3, "bridge2": 2},
        "bridge1": {"banana": 4},
        "bridge2": {"banana": 1, "orange": 2},
        "banana": {},
    }
    graph._build_full_index()
    bridges = graph.get_bridge_words("banana", "apple")
    assert bridges == []