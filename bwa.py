"""
Burrows-Wheeler Alignment
---------------------------------------------
This is a very simple implementation of a Burrows-Wheeler Aligner for indexing and sequence alignment.

Differences between this code and the real BWA algorithm:
1. It does not use array D to estimate the lower bound of the number of differences.
2. It does take insertion and deletion into consideration, only mismatch.
3. It does use a difference score to ignoring worse results.
4. It does reduce the required operating memory by storing small fractions of Occ and SA.

Author: Zell Wu
Date: 4/2/2023
"""

import numpy as np


class BWA:
    """ A Burrows-Wheeler Alignment class. """

    def __init__(self, reference: str):
        """ Initiation """
        self.ref = reference + "$"
        self.alphabet = sorted(['a', 'g', 't', 'c'])

    def suffix_array(self):
        """ Get the suffix array of the reference. """
        # List the all rotations with the reference input.
        rotations = []
        for i in range(len(self.ref)):
            rotations.append(self.ref[i:] + self.ref[:i])
        # Sort the rotations alphabetically
        sorted_rotations = sorted(rotations)
        # Gain the initial index of the sorted rotations
        initial_indices = np.argsort(rotations)
        return list(initial_indices), sorted_rotations

    def bwt(self):
        """ Get the Burrows–Wheeler transform array. """
        # Keep the last column in a sequence
        last = ''
        _, sorted_rotations = self.suffix_array()
        for seq in sorted_rotations:
            last += seq[-1]
        return last

    def Occ(self):
        """ Get the Occ matrix.
            Occ(c, k) is the number of occurrences of character c in the prefix L[1..k].
        """
        # Gain the bwt last column
        last_column = self.bwt()
        # Initiation
        totals = {k: 0 for k in "".join(set(last_column))}
        tally_matrix = {k: [] for k in "".join(set(last_column))}
        # Get the Occ matrix
        for i in last_column:
            totals[i] += 1
            for j in tally_matrix.keys():
                if i != j and tally_matrix[j]:
                    tally_matrix[j].append(tally_matrix[j][-1])
                elif i == j:
                    tally_matrix[j].append(totals[i])
                else:
                    tally_matrix[j].append(0)
        return tally_matrix, totals

    def C(self):
        """ Get the C table.
            C[c] is a table that, for each character c in the alphabet,
            contains the number of occurrences of lexically smaller characters in the text.
        """
        first = {}
        totc = 0
        for i, count in sorted(self.Occ()[1].items()):
            first[i] = totc
            totc += count
        return first

    def lf(self, c, i):
        """ The i-th occurrence of character ‘c’ in last is the same text character
            as the i-th occurrence of ‘c’ in the first.
        """
        if i < 0:
            return 0
        Occ = self.Occ()[0]
        first = self.C()
        return first[c] + Occ[c][i] - 1

    def exact_match(self, read):
        """ exact match - no indels or mismatches allowed. """
        # Get the initial low, high values
        last = self.bwt()
        low, high = last.find(read[-1]), last.rfind(read[-1])
        # Iteratively calculate low, high values
        i = len(read) - 1
        while low <= high and i >= 0:
            low = self.lf(read[i], low-1) + 1
            high = self.lf(read[i], high)
            i -= 1
        return self.suffix_array()[0][low: high+1]

    def inexact_recursion(self, low, high, mismatch_left, read, index):
        """ Recursion function for inexact match. """
        # recursion out
        # stop condition 1: entire read has been matched
        if index <= 0:
            return [(low, high)]
        # stop condition 2: the reference not contained the substrings
        if low > high:
            return []
        matches = []
        next_character = read[index-1]
        for c in self.alphabet:
            low_ = self.lf(c, low-1) + 1
            high_ = self.lf(c, high)
            # if the substring was found
            if low_ <= high_:
                # exact match
                if c == next_character:
                    matches.extend(self.inexact_recursion(low_, high_, mismatch_left, read, index-1))
                # mismatch
                elif mismatch_left > 0:
                    matches.extend(self.inexact_recursion(low_, high_, mismatch_left-1, read, index-1))
        return matches

    def inexact_match(self, read, mismatch=1):
        """ inexact match - only mismatches allowed. """
        return [self.suffix_array()[0][match[0]] for match in self.inexact_recursion(1, len(self.bwt())-1, mismatch, read, len(read))]


if __name__ == "__main__":
    ref = 'atgcgtaatgccgtcgatcg'
    read = 'gta'
    bwa = BWA(ref)
    matches = bwa.inexact_match(read, mismatch=1)

