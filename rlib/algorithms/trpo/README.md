# TRPO

- TRPO is a policy gradient algorithm, which means it optimizes the policy gradient.
- TRPO is an on-policy algorithm.
- TRPO can be used for environments with either discrete or continuous action spaces.

TRPO updates policies by taking the largest step possible to improve performance, while satisfying a special constraint on how close the new and old policies are allowed to be. The constraint is expressed in terms of KL-Divergence, a measure of (something like, but not exactly) distance between probability distributions.

TRPO avoids the issue of VPG where a single bad step can collapse the policy performance, making it impossible to use large step sizes and leading to bad sample efficiency (since it is an on-policy method, more steps means it will need to generate more samples).