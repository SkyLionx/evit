from IPython import get_ipython
import datetime

def is_using_colab() -> bool:
    return "google.colab" in str(get_ipython())

def format_current_date() -> str:
    today = datetime.datetime.today()
    return today.strftime("%Y-%m-%d %H-%M-%S")
