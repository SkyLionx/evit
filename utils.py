from IPython import get_ipython


def is_using_colab() -> bool:
    return "google.colab" in str(get_ipython())
