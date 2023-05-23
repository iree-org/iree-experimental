import subprocess
import json
import re


def get_python_environment_info():
  """ Returns a dictionary of package versions in the python virtual environment."""
  output = subprocess.check_output(["pip", "list"]).decode("utf-8")
  # The first few lines are the table headers so we remove that.
  output = output[output.rindex("---\n") + 4:]
  output = output.split("\n")
  package_dict = {}
  for item in output:
    split = re.split("\s+", item)
    if len(split) == 2:
      package_dict[split[0]] = split[1]
  return package_dict
