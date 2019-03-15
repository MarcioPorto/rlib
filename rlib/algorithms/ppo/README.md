# PPO: Proximal Policy Optimization

Aims to improve upon the REINFORCE/VPG algorithm. More specifically, it makes the update process more efficient, the gradient estimate `g` less noisy, and has clearer credit assignment.

## Noise Reduction

In VPG, the policy is optimized by maximizing the average rewards using stochastic gradient ascent. The gradient is given by an average of rewards over all the possible trajectories. However, in VPG, not all trajectories are sampled since that is too computationally expensive. Often times, only a single trajectory is sampled, and by chance, that trajectory may not contain much useful information about the policy we are trying to learn. The hope is that after training for a long time, the effect of such bad samples is minimized.

The easy solution here is to simply collect more (N) trajectories, especially using distributed computing to do so in parallel. Another advantage is that now we can look at the distribution of rewards across these multiple trajectories. Learning can then be improved by normalizing the rewards, where \muμ is the mean, and \sigmaσ the standard deviation. This technique is also used in other domains, such as image classification to improve learning. The effect is that we ensure that gradient ascent steps won't be too large or small, while also picking half the actions to encourage and discourage.

## Credit Assignment

At time-step `t`, we can divide the rewards between rewards from the past (t-1) and future (starting at t). Assuming we are dealing with a Markov process, the past rewards should have no influence on the policy gradient `g`, as actions at time-step `t` can only affect future rewards.

## Importance Sampling

Improves the efficiency of policy-based methods by recycling old trajectories. It works by modifying them so they are representative of the new policy.

By chance, the same trajectory generated using `theta`, might be generated using `theta'`, but with different probabilities. By adding a re-weighting factor `P(tau; theta') / P(tau; theta')`, we can modify old trajectories for computing averages for the new policy. This same trick is used across statistics, in unbiased surveys and voting predictions.

In practice the result of this re-weighting factor can easily be close to 0 or infinity. For this reason, we always want to make sure that the re-weighting factor is that far from 1 when using importance sampling.

## The Surrogate Function

When calculating `g` for `pi{theta'}`, and you only have trajectories collected from a previous policy `pi{theta}`, simply add a re-weighting factor to the formula for `g`.

This algorithm is called proximal policy because when the two policies are close enough to each other, certain factors can be ignored because they end up being close to 1.

*Add formula*

If we keep reusing old trajectories and updating our policy, at some point the new policy might become different enough from the old one, so that all the approximations we made could become invalid.

## Clipping Policy Updates

The surrogate function might be good at approximating rewards in the short-term, but in the long-run that estimation deteriorates, which can lead to a really bad policy. Clipping (flattening) the surrogate function at some value helps deal with this problem. In a sense, you get a more conservative update procedure.

## Essence

Compute the clipped surrogate function, and perform updates multiple times using gradient ascent on the clipped surrogate function.