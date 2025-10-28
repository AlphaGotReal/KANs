# KANs
an implementation of KANs in pytorch, to present to class for 5 marks

---

# Theory (Intro)
Explaining the [paper](https://arxiv.org/pdf/2404.19756)

## Kolmogorov Arnold Representation Theorem

### Theorem 
This theorem states that every multivariate function (function with multiple inputs but a single output) can be expressed as a two layered sum of single variable functions.

Let $$f: [0, 1]^{n} \rarr \mathbb{R}$$
then $$f(x_1, x_2, .., x_n) = \sum_{q=1}^{2n+1} \phi_q (\sum_{p=1}^{n} \psi_{p,q}(x_p))$$
where $$\phi_p: \mathbb{R} \rarr \mathbb{R}$$ $$\psi_{p,q}: [0, 1] \rarr \mathbb{R} $$

## Kolmogorov Arnold Networks

Now the neural network is laid such that these $$\phi_p \space \&\space \psi_{p,q}$$
are learnable functions. In the paper they are defined are splines where the control points are learned.

## Innovation

The possiblity of using Kolmogorov Arnold representation theorem to build a neural network has been studied, however most work has stuck to using a two layered system. The innovation lies is using multiple layers to achieve better and faster convergenece. 

(source: the paper)
"Despite their elegant mathematical interpretation, KANs are nothing more than combinations of
splines and MLPs, leveraging their respective strengths and avoiding their respective weaknesse" 

# Paper

