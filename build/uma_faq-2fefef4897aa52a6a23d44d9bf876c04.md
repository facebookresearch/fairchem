# UMA FAQ

This page will be updated with FAQ questions as we get them. Please feel free to add PRs if there are questions/explanations that you think others would find helpful!

:::{admonition} How do I choose the right task?
:class: dropdown

Choose your task based on your application domain:

- **omol**: For molecules, biology, organic chemistry, pharmaceuticals
- **omat**: For inorganic materials, solar cells, alloys, superconductors
- **oc20**: For heterogeneous catalysis, fuel cells, energy conversion
- **odac**: For MOFs and direct air capture applications
- **omc**: For molecular crystals, organic electronics
:::

:::{admonition} Why am I getting a 401 error when loading the model?
:class: dropdown

You need to:
1. Create a HuggingFace account
2. Request access to the [UMA model](https://huggingface.co/facebook/UMA)
3. Create a token with read access at <https://huggingface.co/settings/tokens/>
4. Login with `huggingface-cli login` or set `HF_TOKEN` environment variable
:::

:::{admonition} Can I use UMA for periodic systems with the omol task?
:class: dropdown

The `omol` task was trained on aperiodic (molecular) data, so periodic systems should be treated with caution. For periodic materials, consider using `omat` or `oc20` depending on your application.
:::

:::{admonition} How do I specify charge and spin for molecules?
:class: dropdown

For the `omol` task, set charge and spin in the `atoms.info` dictionary:

```python
atoms.info.update({"spin": 1, "charge": 0})
```

Other tasks (`omat`, `oc20`, `odac`, `omc`) expect `charge=0` and `spin=0`.
:::
