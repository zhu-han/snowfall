# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import (
    Iterable,
    List,
)

import torch
import k2
import time

def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    '''Build CTC topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FSA that converts repeated tokens to a single token.
    '''
    num_states = len(tokens)
    final_state = num_states
    rules = ''
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                rules += f'{i} {i} {tokens[i]} 0 0.0\n'
            else:
                rules += f'{i} {j} {tokens[j]} {tokens[j]} 0.0\n'
        rules += f'{i} {final_state} -1 -1 0.0\n'
    rules += f'{final_state}'
    ans = k2.Fsa.from_str(rules)
    return ans


class CtcTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov = oov
        ctc_topo = build_ctc_topo(list(phones._id2sym.keys()))
        self.ctc_topo = k2.arc_sort(ctc_topo)

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]

        fsa = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = k2.compose(self.ctc_topo, decoding_graph)
        decoding_graph = k2.connect(decoding_graph)
        return decoding_graph

class LmCtcTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 G: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        G:
          Its labels are words.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)
        
        if G.properties & k2.fsa_properties.ARC_SORTED != 0:
            G = k2.arc_sort(G)

        assert oov in words

        self.L_inv = L_inv
        self.G = G
        self.phones = phones
        self.words = words
        self.oov = oov
        ctc_topo = build_ctc_topo(list(phones._id2sym.keys()))
        self.ctc_topo = k2.arc_sort(ctc_topo)

    def compile(self, texts: Iterable[str], intersect_grammer: bool = True) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text, intersect_grammer) for text in texts])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str, intersect_grammer: bool = True) -> k2.Fsa:
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]

        fsa = k2.linear_fsa(word_ids)

        if intersect_grammer:
          decoding_graph = k2.intersect(fsa, self.G)
        else:
          decoding_graph = fsa
        assert fsa.shape[0] == decoding_graph.shape[0], "ERROR: transcripts fsa length changed after composition with G: {fsa.shape[0]} -> {decoding_graph.shape[0]}. word_id: {word_ids}, tokens: {tokens}"
        decoding_graph = k2.connect(k2.intersect(decoding_graph, self.L_inv)).invert_()
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = k2.compose(self.ctc_topo, decoding_graph)
        decoding_graph = k2.connect(decoding_graph)
        return decoding_graph

class CharCtcTrainingGraphCompiler(object):

    def __init__(self,
                 chars: k2.SymbolTable,
                 oov: str = '<UNK>',
                 space: str = '<space>'):
        '''
        Args:
          chars:
            The char symbol table.
        oov:
          Out of vocabulary word.
        '''
        assert oov in chars
        assert space in chars

        self.chars = chars
        self.oov = oov
        self.space = space
        ctc_topo = build_ctc_topo(list(chars._id2sym.keys()))
        self.ctc_topo = k2.arc_sort(ctc_topo)
        # self.draw = 0

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        text = [token if token != ' ' else self.space for token in text]
        tokens = (token if token in self.chars else self.oov
                  for token in text)
        
        char_ids = [self.chars[token] for token in tokens]

        # if len(char_ids) < 10:
        #   self.draw += 1

        fsa = k2.linear_fsa(char_ids)
        fsa.aux_labels = torch.tensor(char_ids + [-1], dtype=torch.int32)
        # fsa = build_fst_topo(char_ids)
        fsa = k2.arc_sort(fsa)
        # if self.draw == 1:
        #   fsa.symbols = self.chars
        #   fsa.aux_symbols = self.chars
        #   fsa.draw("linear_char.svg")
        decoding_graph = k2.compose(self.ctc_topo, fsa)
        decoding_graph = k2.connect(decoding_graph)

        # decoding_graph = build_composed_ctc_topo(char_ids)
        decoding_graph = k2.arc_sort(decoding_graph)
        # if self.draw == 1:
        #   decoding_graph.symbols = self.chars
        #   decoding_graph.aux_symbols = self.chars
        #   decoding_graph.draw("char_decoding_graph.svg")
        # if self.draw == 1:
        #   raise RuntimeError("Graph has been draw")
        return decoding_graph

def build_fst_topo(tokens: List[int]) -> k2.Fsa:
    '''Build fst topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FSA that 
    '''
    num_states = len(tokens)
    final_state = num_states + 1
    rules = ''
    for i in range(num_states):
        rules += f'{i} {i+1} {tokens[i]} {tokens[i]} 0.0\n'
    rules += f'{num_states} {final_state} -1 -1 0.0\n'
    rules += f'{final_state}'
    ans = k2.Fsa.from_str(rules)
    return ans

def build_composed_ctc_topo(tokens: List[int]) -> k2.Fsa:
    '''Build composed ctc topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FSA that 
    '''

    num_tokens = len(tokens) 
    final_state = num_tokens * 2 + 1
    rules = ''
    rules += f'0 0 0 0 0.0\n'
    rules += f'0 1 {tokens[0]} {tokens[0]} 0.0\n'
    rules += f'1 1 {tokens[0]} 0 0.0\n'
    for i in range(0, num_tokens - 1):
      rules += f'{2*i+1} {2*i+2} 0 0 0.0\n'
      rules += f'{2*i+1} {2*i+3} {tokens[i+1]} {tokens[i+1]} 0.0\n'
      rules += f'{2*i+2} {2*i+2} 0 0 0.0\n'
      rules += f'{2*i+2} {2*i+3} {tokens[i+1]} {tokens[i+1]} 0.0\n'
      rules += f'{2*i+3} {2*i+3} {tokens[i+1]} 0 0.0\n'
    rules += f'{2*num_tokens-1} {2*num_tokens} 0 0 0.0\n'
    rules += f'{2*num_tokens-1} {2*num_tokens+1} -1 -1 0.0\n'
    rules += f'{2*num_tokens} {2*num_tokens} 0 0 0.0\n'
    rules += f'{2*num_tokens} {2*num_tokens+1} -1 -1 0.0\n'
    rules += f'{final_state}'
    ans = k2.Fsa.from_str(rules)
    return ans
    

