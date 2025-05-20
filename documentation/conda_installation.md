# Managing Your LLM Evaluation Framework with Conda

This guide explains how to convert your current Python environment (managed by `requirements.txt`) into a Conda-managed environment. This can offer better dependency resolution, especially for complex packages, and provide a more consistent setup experience for users. We'll also discuss strategies to mitigate slow package installations.

## 1. Why Use Conda?

* **Robust Dependency Management:** Conda is excellent at handling complex dependencies, especially for packages that involve compiled code (like NumPy, SciPy, PyTorch). It often resolves conflicts more effectively than pip alone.
* **Environment Isolation:** Like Python's `venv`, Conda creates isolated environments, preventing package conflicts between projects.
* **Cross-Platform:** Conda environments are generally more portable across different operating systems.
* **Channel Management:** Conda allows you to specify "channels" (repositories) from which to download packages. Popular channels like `conda-forge` and specific channels like `pytorch` often provide optimized or more up-to-date versions of packages.

## 2. Creating a Conda `environment.yml` File

The Conda equivalent of a `requirements.txt` file is typically an `environment.yml` file. This YAML file specifies the environment's name, channels, and dependencies.

**Steps to Create `environment.yml`:**

1.  **Identify Core Dependencies:** Review your `requirements.txt`.
2.  **Find Conda Equivalents:** Many packages are available directly through Conda channels. Some might have slightly different names or are best installed from specific channels (e.g., `pytorch` from the `pytorch` channel).
3.  **Structure the YAML:**

    ```yaml
    name: llm-eval-env  # Choose a name for your environment
    channels:
      - defaults
      - conda-forge
      - pytorch         # For PyTorch and related packages like sentence-transformers
      - anaconda        # Often a good fallback
    dependencies:
      - python=3.9      # Specify your desired Python version
      # Core Data Science & NLP
      - pandas>=1.5.0,<3.0.0
      - numpy>=1.20.0,<2.0.0
      - matplotlib
      - scikit-learn>=1.0.0,<2.0.0
      - nltk>=3.6.0,<4.0.0
      - openpyxl>=3.0.0,<4.0.0 # For Excel
      - tabulate                # For report generation (CLI)
      # Streamlit and Plotting
      - streamlit>=1.28.0,<2.0.0
      - plotly>=5.10.0,<6.0.0
      # Specific NLP Metrics Libraries
      - rouge-score>=0.1.0,<0.2.0
      - sacrebleu>=2.0.0,<3.0.0 # Note: NLTK's BLEU is also an option but SacreBLEU is often preferred
      # PyTorch and Sentence Transformers
      # sentence-transformers will pull in PyTorch. Explicitly listing PyTorch can help.
      # Check the latest PyTorch installation command for your OS/CUDA version from pytorch.org
      # For CPU-only:
      - pytorch::pytorch # Use the pytorch channel
      - torchvision     # Often a companion to PyTorch
      - torchaudio      # Often a companion to PyTorch
      - sentence-transformers # This should now resolve PyTorch from the pytorch channel
      # Pip dependencies (for packages not readily available on Conda or for specific versions)
      # Conda will run pip after installing conda packages.
      # - pip:
      #   - some-pip-only-package==1.0 
      #   - -r ./requirements_pip_only.txt # If you have a separate file for pip-only
    ```

**Explanation of the `environment.yml`:**

* **`name`**: The name your users will use to activate the environment (e.g., `conda activate llm-eval-env`).
* **`channels`**: The order matters. Conda will search for packages in this order.
    * `defaults`: Conda's default channels.
    * `conda-forge`: A community-driven channel with a vast number of packages. It's often recommended to put this high in the list.
    * `pytorch`: Essential for getting the correct PyTorch build, especially if CUDA is involved (though this example is CPU-focused).
    * `anaconda`: The main channel from Anaconda, Inc.
* **`dependencies`**:
    * List Conda packages directly. You can specify versions like in `requirements.txt`.
    * **`pip` section (optional):** If some packages are only available via pip, or you need a very specific pip version not on Conda, list them under a `pip:` key. Conda will install these using pip after the Conda packages are set up.

**Important Considerations:**

* **PyTorch:** The `sentence-transformers` library depends on PyTorch. Installing PyTorch correctly (especially with GPU support if needed) is crucial. The `pytorch` channel is the official source. The example above shows a CPU-only setup. If GPU is needed, users would typically get the specific Conda install command from [pytorch.org](https://pytorch.org/).
* **Version Specificity:** Be as specific as necessary with versions to ensure reproducibility. However, overly strict versions can sometimes lead to resolution issues. Your current `requirements.txt` uses `>=` and `<` which is a good balance.
* **Testing:** After creating your `environment.yml`, thoroughly test the environment creation and application functionality on a clean system if possible.

## 3. How Users Will Use the `environment.yml`

Provide these instructions to your users:

1.  **Install Conda:** If they don't have it, they need to install Miniconda (a minimal installer for Conda) or Anaconda Distribution from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2.  **Download/Obtain Files:** Ensure they have the `environment.yml` file and the rest of your framework's source code.
3.  **Create and Activate Environment:**
    Open their terminal or Anaconda Prompt, navigate to the directory containing `environment.yml`, and run:
    ```bash
    conda env create -f environment.yml
    ```
    This will create the environment with the specified name (e.g., `llm-eval-env`). After it's created, they activate it:
    ```bash
    conda activate llm-eval-env
    ```
4.  **NLTK Data (Still Required):**
    The NLTK data download step is still necessary after activating the Conda environment:
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    ```
5.  **Run the Application:**
    They can then run the Streamlit app or CLI commands as usual from within the activated Conda environment:
    ```bash
    streamlit run streamlit_app.py
    # OR
    python main.py --input-file ...
    ```

## 4. Addressing Slow Package Installations

Slow installations can be frustrating. Here are strategies to mitigate this:

1.  **Use Mamba (Highly Recommended for Speed):**
    * Mamba is a re-implementation of the Conda package manager in C++ and is significantly faster at dependency solving and downloading.
    * **Installation:** `conda install -n base -c conda-forge mamba` (installs Mamba into your base Conda environment).
    * **Usage:** Users can then replace `conda` with `mamba` for environment creation:
        ```bash
        mamba env create -f environment.yml
        mamba activate llm-eval-env
        ```
    This is often the single most effective way to speed up Conda environment setup.

2.  **Optimize Channel Order:**
    Placing `conda-forge` high in your channel list can sometimes improve resolution speed as it has a very comprehensive set of packages. The order provided in the example `environment.yml` is a good starting point.

3.  **Specify Python Version:**
    Clearly specifying the Python version in `environment.yml` helps Conda narrow down the search space for compatible packages.

4.  **Pre-built Binaries:**
    Conda excels at providing pre-compiled binaries for many platforms. This avoids lengthy compilation times that can occur with `pip` for packages like NumPy or SciPy if a binary wheel isn't available for the user's specific system/Python version.

5.  **Large Packages (PyTorch/Sentence-Transformers):**
    * Packages like PyTorch are inherently large. `sentence-transformers` pulls in PyTorch. While Mamba and the `pytorch` channel help, the download itself will take time depending on network speed.
    * If users need GPU support, the PyTorch download will be even larger. Providing clear instructions for CPU vs. GPU (and linking to [pytorch.org](https://pytorch.org/) for the exact command) is helpful.

6.  **Network Connection:**
    A slow or unstable internet connection will naturally lead to slow downloads. Advise users to ensure they have a stable connection.

7.  **Conda Configuration (Advanced):**
    Users can sometimes speed up Conda by configuring settings like `channel_priority` (e.g., to `strict`), but this is more advanced and usually not needed if Mamba is used.

8.  **Minimize `pip` Dependencies:**
    Whenever possible, find Conda equivalents for packages. Relying heavily on `pip` within a Conda environment can sometimes slow down the environment creation or lead to less robust dependency handling, as Conda first resolves its packages, then pip resolves its own.

By providing an `environment.yml` and recommending Mamba, you can significantly improve the installation experience for your users.
