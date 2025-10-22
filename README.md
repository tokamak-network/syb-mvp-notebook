# syb-jupyter-notebooks

We implement Python code for demo and testing purposes of Sybil-resistant scoring algorithms. We collect the descriptions of the algorithms in the corresponding Binder Jupyter Notebooks.

## Notebooks & Scoring algorithms:

We implement 6 notebooks:
- **Interactive User Experience**: [SYB Network Experience](syb_user_experience.ipynb) - Interactive notebook with a Remix IDE-style interface for experiencing the sybil resistance protocol. Connect as a user, interact with contract functions (deposit, withdraw, vouch, etc.), and observe how actions affect network scores and trust relationships in real-time.
- The main one where we discuss and compare four algorithms: [Comparison Notebook](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/af8dec45a24bf7fd5858923a39de702491239af2?urlpath=lab%2Ftree%2Finitial_notebook.ipynb).
- Four scoring algorithms:
    - [Random walk scoring algorithm](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/70d5992921de4e526111b42821bedcff911add29?urlpath=lab%2Ftree%2Frandom_walk_scoring_algorithm.ipynb)
    - [Equal split scoring algorithm](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/70d5992921de4e526111b42821bedcff911add29?urlpath=lab%2Ftree%2Fequal_split_algorithm.ipynb)
    - [Pagerank scoring algorithm](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/70d5992921de4e526111b42821bedcff911add29?urlpath=lab%2Ftree%2Fpagerank_algorithm.ipynb)
    - [Argmax scoring algorithm](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/af8dec45a24bf7fd5858923a39de702491239af2?urlpath=lab%2Ftree%2Fargmax_algorithm.ipynb)