import heapq
import numpy as np

class Huffman:
    def __init__(self):
        self.phead = None
        self.encode_dictionary = {}  # to store huffman encodings of words
        self.assis_set = set()  # to store non-leaf node encodings

    def create_tree(self, freq_list):
        """
        Generate huffman tree
        """
        p = PriorityQueue()
        for item in freq_list:
            p.push(item[0], item[1])
        while p.length() > 1:
            right, left = p.pop(), p.pop()  # left child value > right child value
            node = Node(left, right)
            p.push(left[0] + right[0], node)
        self.phead = p.pop()

    def huff_encode(self, node):
        self.encoding(node)  # get all encodings and encoding dictionary
        self.assis_set -= set(self.encode_dictionary.values())  # get non-leaf no huffman encodings

    def encoding(self, node, prefix=""):
        """
        Huffman encoding
        :node tuple:(freq, index(unimportant), HuffmanNode)
        :param node: Node
        :param prefix: huffman code
        :return:
        """
        # add leaf node and non-leaf node huffman encoding
        self.assis_set.add(prefix)

        # in-order travel
        if isinstance(node[2], Node):
            if node[2].left:
                self.encoding(node[2].left, prefix + "1")  # left child encodes 1
            if node[2].right:
                self.encoding(node[2].right, prefix + "0")  # right child encodes 0
        else:
            self.encode_dictionary[node[2]] = prefix  # when arrive at leaf node, encoding huffman code for word to dictionary

class PriorityQueue:
    """
    PriorityQueue, using assis index to avoid reduplicative value when comparing
    """

    def __init__(self):
        self._index = 0
        self.queue = []

    def length(self):
        return len(self.queue)

    def push(self, priority, val):
        heapq.heappush(self.queue, (priority, self._index, val))
        self._index += 1

    def pop(self):
        return heapq.heappop(self.queue)


class Node:
    """
    Node object for Huffman
    """

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right


def sigmoid(x):
    """
    Sigmoid function
    """
    # efficient by looking up table
    if x <= -6:
        return 0
    elif x >= 6:
        return 1
    else:
        return 1 / (1 + np.exp(-x))

def dynamic_alpha(alpha, i, length, alpha_min):
    """
    Modify learning rate dynamic
    """
    new_alpha = alpha * (1 - i / (length + 1))
    if new_alpha < alpha_min:
        alpha = alpha_min
    else:
        alpha = new_alpha
    return alpha