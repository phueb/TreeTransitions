<div align="center">
 <img src="images/logo.png" width="250"> 
</div>


## About

This repository contains research that tests the hypothesis that the RNN language model learns via progressive differentiation of hierarchically structured input. 

The input data consists of artificially generated sequences of two items, with the form `X Y`. 
Items in X and Y are disjoint. 
The relationship between X-words is hierarchical because the Y-words they tend to co-occur with provide distributional evidence for a hierarchical category structure among X-words.
Specifically, there are 32 categories of X-words, and each is nested in 16 superordinate categories, and each of these, in turn, are nested in 8 superordinate categories, and so on. 


## Compatibility

Developed on Ubuntu 18.04 using Python3.7