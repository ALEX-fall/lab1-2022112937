import re
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter
import math
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx

class TextGraph:
    def __init__(self):
        self.graph = defaultdict(dict)  # 邻接表：{word: {neighbor: weight}}
        self.pr_values = None
        self.documents = []  # 存储分句后的文档
        self.word_freq = Counter()  # 词频统计

    def add_edge(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        self.graph[word1][word2] = self.graph[word1].get(word2, 0) + 1

    def build_graph(self, text):
        """基础建图方法"""
        # 分句
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        self.documents = sentences

        # 构建图和统计词频
        for sentence in sentences:
            words = re.sub(r'[^a-zA-Z\s]', ' ', sentence).lower().split()
            self.word_freq.update(words)
            for i in range(len(words)-1):
                self.add_edge(words[i], words[i+1])

class TextGraphOptimized(TextGraph):
    def __init__(self):
        super().__init__()
        self.node_index = {}    # 节点到索引的映射
        self.index_node = []    # 索引到节点的映射
        self.tfidf_values = {}  # 存储TFIDF值

    def build_graph(self, text):
        """增强版建图方法"""
        super().build_graph(text)
        self._build_full_index()
        self._calculate_tfidf()

    def _build_full_index(self):
        """建立完整节点索引"""
        all_nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_nodes.update(neighbors.keys())
        self.index_node = list(all_nodes)
        self.node_index = {node: i for i, node in enumerate(self.index_node)}

    def _calculate_tfidf(self):
        """计算每个词的TFIDF值"""
        total_docs = len(self.documents)
        
        # 计算IDF
        doc_freq = defaultdict(int)
        for doc in self.documents:
            words = set(re.sub(r'[^a-zA-Z\s]', ' ', doc).lower().split())
            for word in words:
                doc_freq[word] += 1
        
        # 计算TFIDF
        for word in self.graph.keys():
            tf = self.word_freq[word]
            idf = math.log(total_docs / (1 + doc_freq[word]))
            self.tfidf_values[word] = tf * idf
        
        # 归一化TFIDF值
        if self.tfidf_values:
            max_tfidf = max(self.tfidf_values.values())
            for word in self.tfidf_values:
                self.tfidf_values[word] /= max_tfidf

    def calculate_pagerank(self, d=0.85, max_iter=100, tol=1e-6):
        """使用TFIDF初始化的PageRank算法实现"""
        # 构建NetworkX图对象
        G = nx.DiGraph()
        for src, neighbors in self.graph.items():
            total_weight = sum(neighbors.values())
            for dst, weight in neighbors.items():
                G.add_edge(src, dst, weight=weight/total_weight)
        
        # 使用TFIDF值作为初始PR值
        personalization = {}
        for node in G.nodes():
            personalization[node] = self.tfidf_values.get(node, 0.1)
        
        # 计算PageRank
        self.pr_values = nx.pagerank(G, 
                                   alpha=d, 
                                   max_iter=max_iter, 
                                   tol=tol,
                                   personalization=personalization)
        return self.pr_values

    def get_pagerank(self, word):
        """查询指定单词的PageRank值"""
        if not self.pr_values:
            self.calculate_pagerank()
        return self.pr_values.get(word.lower(), None)

    def get_bridge_words(self, word1, word2):
        """桥接词查询实现"""
        word1 = word1.lower()
        word2 = word2.lower()
        
        if word1 not in self.node_index or word2 not in self.node_index:
            return None
        
        bridges = []
        successors = self.graph.get(word1, {}).keys()
        
        for candidate in successors:
            if word2 in self.graph.get(candidate, {}):
                bridges.append(candidate)
        
        return bridges

    def generate_new_text(self, input_text):
        """生成新文本实现"""
        words = re.sub(r'[^a-zA-Z\s]', ' ', input_text).lower().split()
        new_text = []
        for i in range(len(words)-1):
            new_text.append(words[i])
            bridges = self.get_bridge_words(words[i], words[i+1])
            if bridges:
                new_text.append(random.choice(bridges))
        new_text.append(words[-1])
        return ' '.join(new_text).capitalize()

    def shortest_path(self, word1, word2=None):
        """使用NetworkX实现的最短路径查找"""
        word1 = word1.lower()
        if word2:
            word2 = word2.lower()
                
        if word1 not in self.graph:
            return "No Path!", None
                
        if word2 and word2 not in self.graph:
            return "No Path!", None
                
        G = nx.DiGraph()
        for src, neighbors in self.graph.items():
            for dst, weight in neighbors.items():
                G.add_edge(src, dst, weight=weight)
                
        paths_data = []
            
        if word2:  
            try:
                all_shortest_paths = list(nx.all_shortest_paths(G, word1, word2, weight='weight'))
                    
                if not all_shortest_paths:
                    return f"No path exists between {word1} and {word2}", None
                        
                for path in all_shortest_paths:
                    length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                    paths_data.append({
                        'path': path,
                        'length': length,
                        'path_str': " → ".join(path)
                    })
                        
                result = "Found following shortest paths:\n"
                for data in paths_data:
                    result += f"Path: {data['path_str']}\n"
                    result += f"Length: {data['length']}\n\n"
                        
            except nx.NetworkXNoPath:
                return f"No path exists between {word1} and {word2}", None
                    
        else: 
            result = f"Shortest paths from {word1} to all other words:\n\n"
            for target in G.nodes():
                if target != word1:
                    try:
                        path = nx.shortest_path(G, word1, target, weight='weight')
                        length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                        paths_data.append({
                            'path': path,
                            'length': length,
                            'path_str': " → ".join(path)
                        })
                    except nx.NetworkXNoPath:
                        continue
                            
            paths_data.sort(key=lambda x: x['length'])
                
            for data in paths_data:
                result += f"To {data['path'][-1]}:\n"
                result += f"Path: {data['path_str']}\n"
                result += f"Length: {data['length']}\n\n"
                    
        return result, paths_data

    def random_walk(self):
        """随机游走实现"""
        if not self.graph:
            return ""
            
        current = random.choice(list(self.graph.keys()))
        visited_edges = set()
        path = [current]
        
        while True:
            neighbors = list(self.graph.get(current, {}).keys())
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            edge = (current, next_node)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(next_node)
            current = next_node
        
        return ' '.join(path)

class GraphApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("文本图谱分析系统")
        self.geometry("1200x800")
        self.graph = TextGraphOptimized()
        self.current_figure = None
        self.current_path = None
        self._create_widgets()

    def _create_widgets(self):
        """创建GUI组件"""
        control_frame = tk.Frame(self, width=220, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        buttons = [
            ("📂 加载文件", self.load_file),
            ("📊 显示图谱", self.show_graph),
            ("🌉 桥接词查询", self.query_bridge_words),
            ("🔄 生成新文本", self.generate_new_text),
            ("🛤️ 最短路径", self.shortest_path),
            ("📈 重要度排名", self.show_pagerank),
            ("🎲 随机游走", self.random_walk),
            ("❓ 帮助", self.show_help)
        ]

        for text, cmd in buttons:
            btn = tk.Button(control_frame, text=text, command=cmd,
                          font=("微软雅黑", 12), bg="#e1e1e1", relief=tk.GROOVE)
            btn.pack(fill=tk.X, pady=5, ipady=3)

        self.result_area = tk.Text(self, wrap=tk.WORD, font=("宋体", 12))
        self.result_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure_frame = tk.Frame(self)
        self.figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def load_file(self):
        """加载文件处理"""
        filepath = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().replace('\n', ' ')
            self.graph.build_graph(text)
            self._show_result(f"✔️ 成功加载文件：{os.path.basename(filepath)}\n"
                             f"▸ 节点数量：{len(self.graph.graph)}")
        except Exception as e:
            messagebox.showerror("错误", f"文件读取失败：{str(e)}")

    def show_graph(self):
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
    
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        G = nx.DiGraph()
        for src, neighbors in self.graph.graph.items():
            for dst, weight in neighbors.items():
                G.add_edge(src, dst, weight=weight)

        self.current_figure = plt.figure(figsize=(10, 8), dpi=100)
        ax = self.current_figure.add_subplot(111)
        pos = nx.spring_layout(G, k=0.15, iterations=50)
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.2)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                             node_size=500, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        if self.current_path:
            colors = ['r', 'b', 'g', 'c', 'm', 'y']
            for idx, path_data in enumerate(self.current_path):
                path = path_data['path']
                color = colors[idx % len(colors)]
                
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                     nodelist=path,
                                     node_color=color,
                                     node_size=700)
                
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(G, pos, ax=ax,
                                     edgelist=path_edges,
                                     edge_color=color,
                                     width=2)

        ax.set_title("Text Graph Visualization")
        ax.axis('off')

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"text_graph_{timestamp}.png"
            self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"保存图片失败: {str(e)}")

        canvas = FigureCanvasTkAgg(self.current_figure, self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def query_bridge_words(self):
        word1 = simpledialog.askstring("桥接词查询", "请输入第一个单词：")
        word2 = simpledialog.askstring("桥接词查询", "请输入第二个单词：")
        if not word1 or not word2:
            return

        bridges = self.graph.get_bridge_words(word1, word2)
        
        if bridges is None:
            result = f"⚠️ 错误：'{word1}' 或 '{word2}' 不在图谱中！"
        elif not bridges:
            result = f"🔍 未找到从 '{word1}' 到 '{word2}' 的桥接词"
        else:
            if len(bridges) == 1:
                bridge_list = bridges[0]
            else:
                bridge_list = "、".join(bridges[:-1]) + f" 和 {bridges[-1]}"
            result = (f"🌉 从 '{word1}' 到 '{word2}' 发现 {len(bridges)} 个桥接词：\n"
                      f"▸ {bridge_list}")
        
        self._show_result(result)

    def generate_new_text(self):
        input_text = simpledialog.askstring("生成新文本", "请输入原始文本：")
        if not input_text:
            return
        
        new_text = self.graph.generate_new_text(input_text)
        self._show_result(f"🔄 生成结果：\n{new_text}")

    def shortest_path(self):
        word1 = simpledialog.askstring("最短路径", "请输入起点单词：")
        word2 = simpledialog.askstring("最短路径", "请输入终点单词（可选）：")
        if not word1:
            return

        result, paths_data = self.graph.shortest_path(word1, word2)
        self.current_path = paths_data
        self._show_result(result)
        self.show_graph()

    def show_pagerank(self):
        """处理PageRank查询"""
        word = simpledialog.askstring("PageRank查询", "请输入要查询的单词：")
        if not word:
            return
        
        word_lower = word.lower()
        
        if word_lower not in self.graph.graph:
            self._show_result(f"⚠️ 单词 '{word}' 未找到！")
            return
        
        pr_value = self.graph.get_pagerank(word_lower)
        
        result = (f"📊 单词 '{word}' 的PageRank值：\n"
                  f"▸ PR值：{pr_value:.6f}\n\n"
                  )
        self._show_result(result)

    def random_walk(self):
        path = self.graph.random_walk()
        with open("random_walk.txt", 'w', encoding='utf-8') as f:
            f.write(path)
        self._show_result(f"🎲 随机游走结果已保存到文件：\n{path}")

    def show_help(self):
        help_text = """📖 使用指南：
1. 点击【加载文件】选择文本文件
2. 【显示图谱】可视化单词关系网络
3. 【桥接词查询】输入两个单词查找连接词
4. 【生成新文本】根据桥接词扩展文本
5. 【最短路径】查询单词间最短路径
6. 【重要度排名】查看单词影响力排名（使用TFIDF初始化的PageRank）
7. 【随机游走】生成随机文本并保存"""
        messagebox.showinfo("系统帮助", help_text)

    def _show_result(self, text):
        self.result_area.config(state=tk.NORMAL)
        self.result_area.delete(1.0, tk.END)
        self.result_area.insert(tk.END, text + "\n")
        self.result_area.config(state=tk.DISABLED)
        self.result_area.see(tk.END)

def main():
    app = GraphApp()
    app.mainloop()

if __name__ == "__main__":
    print("git修改1")
    print("git修改2")
    main()