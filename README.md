# Fightin' Words
Quick implementation of Monroe et al.'s algorithm for comparing languages.

See their paper for more information. The citation is as follows:

```
@article{monroe2008fightin,
  title={Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict},
  author={Monroe, Burt L and Colaresi, Michael P and Quinn, Kevin M},
  journal={Political Analysis},
  volume={16},
  number={4},
  pages={372--403},
  year={2008},
  publisher={SPM-PMSAPSA}
}
```

## The problem and its solution

Say you have two groups of people talking, and you'd like to know
which side is saying what. There are two problems that you might
encounter when attempting this comparison:

1. You have a disproportionate number of language samples from one side, so methods based on raw counts are not going to work.
2. One side uses words that the other side doesn't, and it's not clear whether this occurs because the word is actually used more by one side, or because the word just happens to be rare.

Monroe et al. solve the first problem by examining the usage *rates*
of each word/n-gram, rather than the raw counts. They solve the second
problem by introducing a smoothing [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior on vocabulary items. These two solutions are unified under a single, simple framework, which I have (hopefully) implemented here.

## This code

The function here takes in two lists of strings, one for each language
you'd like to compare, and outputs a list of tuples of the form
`(n-gram, z-score)` where `n-gram` is an n-gram that you might be
interested in and `z-score` is the signed, model-based z-score that
this n-gram is really being used by one side more than the
other. Remember, any value outside of the range `[-1.96, 1.96]` is
technically significant at the 95% level.

However, be wary: this significance is a *model* based statistical
significance, so it only holds if your model holds. Using an
*informative* prior as described in the paper might convince people
that your model is more accurate, and your z-scores more valid :)
