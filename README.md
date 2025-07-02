# Tutorial notebooks - AI4Health Summer School

> 👀​ How to use these notebooks ?
> Go through the notebooks cells by cells, they are not independant and you will need previous cells to run the later ones.
> Many questions are scattered throughout the notebooks. Questions in the first notebook are interesting to get used to WSI manipulation, 
> questions in the second may be a bit broader and open: you are free to solve these questions yourself (what I would advise) or to just copy the answers present in the `answers.py` file.
> Open questions will be adressed orally. Alternatively, you may find the full corrected notebook in the first commit of this repo.

**_IMPORTANT_** First thing: If you have not done it yet, download this https://drive.google.com/file/d/1VN57GS0d-fVQkBlc63UC7jVXxnw4Dw_W/view?usp=sharing and untar it in this folder.
You should now have an `./assets` subfolder.

**_IMPORTANT BIS_** Second, please download the following tar and untar it in your `./assets` folder! 
These are useful data that I forgot to put in the initial tar

LINK : https://drive.google.com/file/d/1pX_Ai2rXVnPhk4vdk8cEqoQWeY57osn7/view?usp=sharing 

# Requirements for the course:

You will need to install a Python environment with the required dependencies - defined in the `pyproject.toml` file.
To download uv, you can use the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

on Linux, or 

```

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 
```
on Windows, or the [astral installer webpage](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Then, install the dependencies in a local virtual environment:

```bash
uv sync
```

The env has been created in `./.venv`.

If you are working with VS Code (or any other IDE, really), you can select the Python kernel you would like to use within the notebook.
This can be selected at the top right of the notebook window in VS Code.
The one you want to use is located here: `.venv/bin/python`.
If it does not appear directly, then try to independently select a Python interpreter for VS Code:

type `Cmd+Shift+P` (or `Ctrl+Shift+P` on Windows), type `Python: select interpreter` and select `.venv/bin/python`.

If you are using Jupyter Lab, then:

```bash
uv run --with jupyter jupyter lab
```

and open `http://localhost:8888/lab` on your browser.
