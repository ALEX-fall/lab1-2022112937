import pytest
import random
from software_main import TextGraphOptimized

@pytest.fixture
def graph():
    return TextGraphOptimized()

def test_no_bridge_words(graph):
    """测试没有桥接词的情况"""
    graph.graph = {
        'hello': {'world': 1},
        'world': {}
    }
    graph._build_full_index()
    input_text = "hello world"
    result = graph.generate_new_text(input_text)
    assert result == "Hello world"

def test_single_bridge_word(graph, monkeypatch):
    """测试存在单个桥接词的情况"""
    monkeypatch.setattr(random, 'choice', lambda x: x[0])
    graph.graph = {
        'hello': {'beautiful': 1},
        'beautiful': {'world': 1},
        'world': {}
    }
    graph._build_full_index()
    input_text = "hello world"
    result = graph.generate_new_text(input_text)
    assert result == "Hello beautiful world"

def test_multiple_bridge_words(graph, monkeypatch):
    """测试存在多个桥接词的情况"""
    monkeypatch.setattr(random, 'choice', lambda x: x[0])
    graph.graph = {
        'data': {'report': 1, 'analysis': 1},
        'report': {'submitted': 1},
        'analysis': {'submitted': 1},
        'submitted': {}
    }
    graph._build_full_index()
    for i in range(10):
        input_text = "data submitted"
        result = graph.generate_new_text(input_text)
        assert result in ["Data report submitted", "Data analysis submitted"]


def test_punctuation_handling(graph, monkeypatch):
    """测试标点符号处理"""
    monkeypatch.setattr(random, 'choice', lambda x: x[0])
    graph.graph = {
        'hello': {'there': 1},
        'there': {'world': 1},
        'world': {}
    }
    graph._build_full_index()
    input_text = "Hello, world!"
    result = graph.generate_new_text(input_text)
    assert result == "Hello there world"
    
# 测试首字母大写及大小写规范化
def test_capitalization(graph, monkeypatch):
    """测试生成文本的首字母大写及规范化"""
    monkeypatch.setattr(random, 'choice', lambda x: x[0])
    graph.graph = {
        'testing': {'bridge': 1},
        'bridge': {'caps': 1},
        'caps': {}
    }
    graph._build_full_index()
    input_text = "testIng CAPS"
    result = graph.generate_new_text(input_text)
    # 验证：首字母大写，其余字母小写，桥接词位置正确
    assert result == "Testing bridge caps"
    
def test_empty_input(graph):
    assert graph.generate_new_text("") == ""


def test_non_alphabetic_input(graph):
    """测试纯非字母字符输入"""
    assert graph.generate_new_text("123456") == ""

def test_Chinese(graph):
    """测试中文字符输入"""
    assert graph.generate_new_text("中文测试") == ""