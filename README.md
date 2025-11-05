# SYB MVP Interactive Notebook

This repository contains an interactive Jupyter Notebook demonstrating the Minimum Viable Product (MVP) of the SYB Sybil Resistant System.

The main notebook, `syb_mvp.ipynb`, provides an interactive tutorial of the `VouchMinimal` scoring algorithm. It allows you to take on the role of a new "User" node, join a simulated network, and interact with other users by vouching and unvouching.

This tool is an interactive demonstration based on the [SYB project](https://syb.tokamak.network) and its [Network Explorer](https://syb.tokamak.network/explorer).

## 🚀 Getting Started

There are two ways to run this interactive demo:

### 1\. Binder (Recommended)

The easiest way to run this notebook is by using Binder, at the following [link](https://mybinder.org/v2/gh/tokamak-network/syb-jupyter-notebooks/64cd5e7f6e690f0883998e572b398e0f4d93623e?urlpath=lab%2Ftree%2Fsyb_mvp.ipynb).


### 2\. Local Installation

To run the notebook on your own machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The `requirements.txt` file lists all necessary dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook syb_mvp.ipynb
    ```

## notebook-demo What it Demonstrates

The `syb_mvp.ipynb` notebook guides you through a scripted, then interactive, session:

  * **Initialization:** A random network of 8 users (Alice, Bob, etc.) is created, and their initial vouches are established.
  * **Joining the Network:** A new, 9th user named "User" is created. This is the node you will control.
  * **Interactive UI:** The notebook launches an `ipywidgets`-based user interface, setting the focus on your "User" node.
  * **Tutorial:** You will see a step-by-step demonstration of:
    1.  'Alice' vouching for your "User" node.
    2.  'Bob' also vouching for your "User" node, changing its rank and score.
    3.  Your "User" node vouching back for 'Alice'.
    4.  'Alice' unvouching for your "User" node, demonstrating network dynamics.
  * **Live Interaction:** After the script, you can use the interactive UI to freely vouch for and unvouch from other users and see the network ranks and scores update in real-time.
  * **Graph Visualization:** The network is visualized using Plotly. Your "User" node is highlighted with a **diamond shape** for easy identification, while all other nodes are circles.

## 📂 Key Files

  * **`syb_mvp.ipynb`**: The main Jupyter Notebook. It contains the markdown explanations, the scripted tutorial, and the code to launch the interactive UI.
  * **`syb_mvp_ui.py`**: This file contains the `SYBMvpUserInterface` class, which builds the interactive UI using `ipywidgets` and Plotly.
  * **`contract_interface_mvp.py`**: A core file containing the Python implementation of the `VouchMinimal` smart contract. It manages node states, handles `vouch`/`unvouch` logic, and computes the ranks and scores.
  * **`plot_utils.py`**: Contains the `show_network_status` and `show_network_graph` helper functions, which are used by the notebook to render the status tables and the Plotly network graph.
  * **`requirements.txt`**: A list of all Python dependencies required to run the notebook, including `networkx`, `plotly`, `ipywidgets`, and `nbformat`.

## 🔗 References

This project is based on the following specifications and resources:

  * **SYB Project Home:** [https://syb.tokamak.network](https://syb.tokamak.network)
  * **SYB Network Explorer:** [https://syb.tokamak.network/explorer](https://syb.tokamak.network/explorer)
  * **MVP Smart Contract (Sepolia):** [https://sepolia.etherscan.io/address/0x02Cb439549AED1A6c8334430A1D5d320685c3E62\#code](https://sepolia.etherscan.io/address/0x02Cb439549AED1A6c8334430A1D5d320685c3E62#code)
  * **MVP Scoring Algorithm Specification:** [https://www.notion.so/tokamak/SYB-MVP-Algorithm-specification-29cd96a400a380c289c0e15aa2ad242f?source=copy\_link](https://www.notion.so/tokamak/SYB-MVP-Algorithm-specification-29cd96a400a380c289c0e15aa2ad242f?source=copy_link)